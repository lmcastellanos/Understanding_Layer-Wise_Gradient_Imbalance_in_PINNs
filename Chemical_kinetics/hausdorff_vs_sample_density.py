#!/usr/bin/env python3
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SAMPLE_COUNTS = [20, 30, 40, 50]
MODEL_EVALUATION_FOLDER = "model_evaluation"
EVALUATION_CASE = "interpolation"
RECONSTRUCTION_MODE = "autoregression"
PREFERRED_HAUSDORFF_FILENAME = "autoregression_hausdorff_distance.csv"
OUTPUT_FOLDER = "hausdorff_samples_comparison"
Y_MIN = 0.0
Y_MAX = 1.0

TRAINING_CASES = {
    "Gradient_clipping": ("Gradient clipping", Path("Gradient_clipping"), "gradient_clipping"),
    "Mass_conservation_1e-9": (r"Mass conservation $\lambda=10^{-9}$", Path("Mass_conservation") / "1e-9", "mass_conservation_1e-9"),
    "Mass_conservation_1e-12": (r"Mass conservation $\lambda=10^{-12}$", Path("Mass_conservation") / "1e-12", "mass_conservation_1e-12"),
    "MSE": ("MSE", Path("MSE"), "mse"),
}

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

def find_hausdorff_csv(base_folder):
    target = base_folder / MODEL_EVALUATION_FOLDER / EVALUATION_CASE / RECONSTRUCTION_MODE
    if not target.is_dir():
        raise FileNotFoundError(f"Expected folder not found: {target}")

    preferred = target / PREFERRED_HAUSDORFF_FILENAME
    if preferred.is_file():
        return preferred

    candidates = [
        p for p in sorted(target.glob("*.csv"))
        if "hausdorff" in p.name.lower() and "autoregression" in p.name.lower()
    ]
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) == 0:
        raise FileNotFoundError(f"No Hausdorff CSV found in {target}")
    raise RuntimeError(f"Multiple Hausdorff CSVs found in {target}: {candidates}")

def load_mean_by_species(csv_path):
    df = pd.read_csv(csv_path)
    unnamed = [c for c in df.columns if str(c).lower().startswith("unnamed:")]
    if unnamed:
        df = df.drop(columns=unnamed)
    num = df.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")
    if num.shape[1] == 0:
        raise ValueError(f"No numeric species columns in {csv_path}")
    return num.mean(axis=0, skipna=True)

def collect_results(root):
    results = {key: {} for key in TRAINING_CASES}
    for n in SAMPLE_COUNTS:
        sample_dir = root / f"{n}_samples"
        if not sample_dir.is_dir():
            raise FileNotFoundError(f"Missing sample folder: {sample_dir}")

        for key, (_label, relpath, _slug) in TRAINING_CASES.items():
            csv_path = find_hausdorff_csv(sample_dir / relpath)
            results[key][n] = load_mean_by_species(csv_path)
            print(f"{n} samples | {key} | {csv_path}")
    return results

def build_dataframe(sample_results):
    species = []
    for n in SAMPLE_COUNTS:
        for s in sample_results[n].index:
            if s not in species:
                species.append(s)

    out = pd.DataFrame(index=species)
    out.index.name = "species"
    for n in SAMPLE_COUNTS:
        out[f"{n} samples"] = sample_results[n].reindex(species)
    return out

def plot_grouped(df, title, output_path):
    species = df.index.tolist()
    labels = df.columns.tolist()
    x = np.arange(len(species))
    n = len(labels)
    width = 0.82 / n

    fig, ax = plt.subplots(figsize=(max(10, 1.25 * len(species)), 6.5))

    for i, label in enumerate(labels):
        offset = (i - (n - 1) / 2) * width
        ax.bar(x + offset, df[label].to_numpy(dtype=float), width=width, label=label)

    ax.set_xlabel("Chemical species")
    ax.set_ylabel("Mean Hausdorff distance")
    #ax.set_title(title + " — autoregressive interpolation")
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_xticks(x)
    ax.set_xticklabels(species, rotation=45, ha="right")
    ax.legend(title="Training dataset", frameon=False)
    ax.grid(axis="y", alpha=0.2)

    fig.tight_layout()
    fig.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close(fig)

def main():
    root = Path(__file__).resolve().parent
    results = collect_results(root)

    outdir = root / OUTPUT_FOLDER
    outdir.mkdir(parents=True, exist_ok=True)

    long_rows = []

    for key, sample_results in results.items():
        label, _relpath, slug = TRAINING_CASES[key]
        df = build_dataframe(sample_results)

        df.to_csv(outdir / f"{slug}_hausdorff_vs_samples.csv")
        plot_grouped(df, label, outdir / f"{slug}_hausdorff_vs_samples.png")

        for n in SAMPLE_COUNTS:
            for species, value in sample_results[n].items():
                long_rows.append({
                    "training_case": key,
                    "samples": n,
                    "species": species,
                    "mean_hausdorff_distance": value,
                })

    pd.DataFrame(long_rows).to_csv(
        outdir / "all_hausdorff_vs_samples.csv",
        index=False,
    )

    print(f"Results saved in: {outdir}")

if __name__ == "__main__":
    main()