#!/usr/bin/env python3

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MASS_CONSERVATION_CONSTANTS = ["1e-9", "1e-12"]

METHOD_FOLDERS = {
    "Gradient clipping": "Gradient_clipping",
    "MSE": "MSE",
}

MASS_FOLDER = "Mass_conservation"
MODEL_EVALUATION_FOLDER = "model_evaluation"
EVALUATION_CASE = "interpolation"
RECONSTRUCTION_MODE = "autoregression"
PREFERRED_HAUSDORFF_FILENAME = "autoregression_hausdorff_distance.csv"

OUTPUT_FOLDER = "hausdorff_comparison"
OUTPUT_PNG = "hausdorff_distance_comparison.png"
OUTPUT_CSV = "hausdorff_distance_comparison.csv"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 600,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def find_hausdorff_csv(base_folder: Path) -> Path:
    target_dir = (
        base_folder
        / MODEL_EVALUATION_FOLDER
        / EVALUATION_CASE
        / RECONSTRUCTION_MODE
    )

    if not target_dir.is_dir():
        raise FileNotFoundError(f"Expected folder not found: {target_dir}")

    preferred = target_dir / PREFERRED_HAUSDORFF_FILENAME
    if preferred.is_file():
        return preferred

    candidates = [
        path for path in sorted(target_dir.glob("*.csv"))
        if "hausdorff" in path.name.lower()
        and "autoregression" in path.name.lower()
    ]

    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) == 0:
        raise FileNotFoundError(
            f"No autoregression Hausdorff-distance CSV found in: {target_dir}"
        )

    raise RuntimeError(
        f"More than one possible Hausdorff CSV found in {target_dir}: "
        + ", ".join(path.name for path in candidates)
    )


def load_mean_hausdorff_by_species(csv_path: Path) -> pd.Series:
    df = pd.read_csv(csv_path)

    if df.empty:
        raise ValueError(f"Hausdorff CSV is empty: {csv_path}")

    unnamed = [
        col for col in df.columns
        if str(col).lower().startswith("unnamed:")
    ]
    if unnamed:
        df = df.drop(columns=unnamed)

    numeric = df.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")

    if numeric.shape[1] == 0:
        raise ValueError(f"No numeric species columns found in: {csv_path}")

    return numeric.mean(axis=0, skipna=True)


def collect_all_results(script_dir: Path):
    results = {}
    source_paths = {}

    for label, folder_name in METHOD_FOLDERS.items():
        method_folder = script_dir / folder_name
        csv_path = find_hausdorff_csv(method_folder)
        results[label] = load_mean_hausdorff_by_species(csv_path)
        source_paths[label] = csv_path

    for constant in MASS_CONSERVATION_CONSTANTS:
        label = f"Mass conservation lambda={constant}"
        method_folder = script_dir / MASS_FOLDER / constant
        csv_path = find_hausdorff_csv(method_folder)
        results[label] = load_mean_hausdorff_by_species(csv_path)
        source_paths[label] = csv_path

    return results, source_paths


def build_comparison_dataframe(results: dict) -> pd.DataFrame:
    species_order = []

    for series in results.values():
        for species in series.index:
            if species not in species_order:
                species_order.append(species)

    comparison = pd.DataFrame(index=species_order)

    for label, series in results.items():
        comparison[label] = series.reindex(species_order)

    comparison.index.name = "species"
    return comparison


def plot_grouped_bars(comparison: pd.DataFrame, output_path: Path):
    species = comparison.index.tolist()
    labels = comparison.columns.tolist()

    x = np.arange(len(species))
    n_series = len(labels)

    group_width = 0.82
    bar_width = group_width / max(n_series, 1)

    fig_width = max(10, 1.25 * len(species))
    fig, ax = plt.subplots(figsize=(fig_width, 6.5))

    for idx, label in enumerate(labels):
        offset = (idx - (n_series - 1) / 2) * bar_width
        values = comparison[label].to_numpy(dtype=float)

        ax.bar(
            x + offset,
            values,
            width=bar_width,
            label=label,
        )

    ax.set_xlabel("Chemical species")
    ax.set_ylabel("Mean Hausdorff distance")
    ax.set_ylim(0, 1)
    #ax.set_title("Autoregressive reconstruction - interpolation")

    ax.set_xticks(x)
    ax.set_xticklabels(species, rotation=45, ha="right")

    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)

    fig.tight_layout()
    fig.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def main():
    script_dir = Path(__file__).resolve().parent

    results, source_paths = collect_all_results(script_dir)

    print("Hausdorff files loaded:")
    for label, path in source_paths.items():
        print(f"  {label}: {path}")

    comparison = build_comparison_dataframe(results)

    output_dir = script_dir / OUTPUT_FOLDER
    output_dir.mkdir(parents=True, exist_ok=True)

    output_csv = output_dir / OUTPUT_CSV
    output_png = output_dir / OUTPUT_PNG

    comparison.to_csv(output_csv)
    plot_grouped_bars(comparison, output_png)

    print(f"Saved table: {output_csv}")
    print(f"Saved plot : {output_png}")


if __name__ == "__main__":
    main()