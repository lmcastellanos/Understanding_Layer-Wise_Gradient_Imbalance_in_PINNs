#!/usr/bin/env python3

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Plot configuration
# ---------------------------------------------------------------------

BINS = 60
SMOOTH_SIGMA = 2.0

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "figure.dpi": 300,
    "lines.linewidth": 1.5,
    "savefig.dpi": 600,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def gaussian_kernel(size: int = 13, sigma: float = 2.0) -> np.ndarray:
    x = np.arange(size) - (size - 1) / 2
    kernel = np.exp(-(x ** 2) / (2 * sigma ** 2))
    kernel /= kernel.sum()
    return kernel


def smooth_density_from_histogram(
    values: np.ndarray,
    bins: int = 60,
    smooth_sigma: float = 2.0
):
    counts, bin_edges = np.histogram(values, bins=bins, density=True)
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    kernel_size = max(7, int(6 * smooth_sigma) | 1)
    kernel = gaussian_kernel(kernel_size, sigma=smooth_sigma)

    smooth_counts = np.convolve(counts, kernel, mode="same")

    return centers, smooth_counts


def sanitize_name(name: str) -> str:
    return (
        str(name)
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
    )


# ---------------------------------------------------------------------
# Automatic model detection
# ---------------------------------------------------------------------

def detect_model(script_dir: Path):
    """
    Detect exactly one final model:

        k_model_0.pth ... k_model_4.pth
        K_model_0.pth ... K_model_4.pth

    Returns
    -------
    model_path : Path
    model_number : int
    """

    candidates = []

    for path in script_dir.iterdir():
        if not path.is_file():
            continue

        match = re.fullmatch(
            r"[Kk]_model_([0-4])\.pth",
            path.name
        )

        if match:
            candidates.append(
                (path, int(match.group(1)))
            )

    if len(candidates) == 0:
        raise FileNotFoundError(
            "No final model found. Expected exactly one file named "
            "k_model_n.pth or K_model_n.pth with n between 0 and 4."
        )

    if len(candidates) > 1:
        names = ", ".join(
            path.name for path, _ in candidates
        )
        raise RuntimeError(
            "More than one final model was found. "
            f"Expected exactly one. Found: {names}"
        )

    model_path, model_number = candidates[0]

    print(
        f"Detected model: {model_path.name} "
        f"(model number = {model_number})"
    )

    return model_path, model_number


# ---------------------------------------------------------------------
# Locate latest gradient CSV belonging to detected model
# ---------------------------------------------------------------------

def find_latest_gradient_csv(
    script_dir: Path,
    model_number: int
):
    """
    Search:

        gradients_by_run/
            run_n/
                epoch_x/
                    gradients_run_n_epoch_x.csv

    and return the CSV corresponding to the largest available epoch.
    """

    run_dir = (
        script_dir
        / "gradients_by_run"
        / f"run_{model_number}"
    )

    if not run_dir.exists():
        raise FileNotFoundError(
            f"Gradient folder not found for model {model_number}: "
            f"{run_dir}"
        )

    candidates = []

    pattern = re.compile(
        rf"gradients_run_{model_number}_epoch_(\d+)\.csv$"
    )

    for csv_path in run_dir.rglob("*.csv"):
        match = pattern.fullmatch(csv_path.name)

        if match:
            epoch = int(match.group(1))
            candidates.append(
                (epoch, csv_path)
            )

    if len(candidates) == 0:
        raise FileNotFoundError(
            f"No gradient CSV found for model {model_number} "
            f"inside {run_dir}"
        )

    candidates.sort(
        key=lambda item: item[0]
    )

    epoch, csv_path = candidates[-1]

    print(
        f"Using gradient CSV: {csv_path}"
    )
    print(
        f"Detected gradient epoch: {epoch}"
    )

    return csv_path, epoch


# ---------------------------------------------------------------------
# Histogram plotting
# ---------------------------------------------------------------------

def save_gradient_histogram(
    values,
    layer_name,
    loss_part,
    output_path: Path,
    bins=BINS,
    smooth_sigma=SMOOTH_SIGMA
):
    """
    Save one publication-style histogram with a smoothed density overlay
    and a compact statistics box.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if values.size == 0:
        print(f"Skipping empty gradient distribution: {output_path}")
        return False

    fig, ax = plt.subplots(figsize=(9, 6))

    # Histogram bars
    ax.hist(
        values,
        bins=bins,
        density=True,
        alpha=0.55,
        label="Gradient distribution",
    )

    # Smoothed histogram density.
    # Skip only if all values are effectively identical.
    if values.size > 1 and not np.allclose(values, values[0]):
        x_smooth, y_smooth = smooth_density_from_histogram(
            values,
            bins=bins,
            smooth_sigma=smooth_sigma,
        )
        ax.plot(
            x_smooth,
            y_smooth,
            linewidth=2.0,
            label="Smoothed density",
        )

    # Display labels. The regularizer plotted here is the SCALED
    # regularizer gradient, but it is intentionally shown as L_mass.
    loss_label = {
        "mse": r"$\mathcal{L}_{MSE}$",
        "scaled_reg": r"$\mathcal{L}_{mass}$",
        "combined": r"$\mathcal{L}_{total}$",
    }.get(loss_part, loss_part)

    title = f"Layer: {layer_name} | Gradients of {loss_label}"

    ax.set_title(title)
    ax.set_xlabel("Gradient value")
    ax.set_ylabel("Density")
    ax.legend()

    # Compact statistics box, matching the reference histogram format.
    stats_text = (
        f"N = {values.size}\n"
        f"mean = {np.mean(values):.3e}\n"
        f"std = {np.std(values):.3e}"
    )

    ax.text(
        0.98,
        0.96,
        stats_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        bbox=dict(boxstyle="round", alpha=0.15),
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=600,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved: {output_path}")
    return True


# ---------------------------------------------------------------------
# Build histograms by layer
# ---------------------------------------------------------------------

def build_histograms(
    df: pd.DataFrame,
    output_root: Path,
    model_number: int,
    epoch: int
):
    required_columns = {
        "run_id",
        "epoch",
        "loss_part",
        "layer",
        "gradient"
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            "Gradient CSV is missing required columns: "
            f"{sorted(missing)}"
        )

    # Only keep rows belonging to the detected final model.
    df = df[
        pd.to_numeric(
            df["run_id"],
            errors="coerce"
        ) == model_number
    ].copy()

    if df.empty:
        raise ValueError(
            f"No gradient data were found for model {model_number}."
        )

    # Prefer the epoch inferred from the filename/folder.
    df = df[
        pd.to_numeric(
            df["epoch"],
            errors="coerce"
        ) == epoch
    ].copy()

    if df.empty:
        raise ValueError(
            f"No gradient rows found at epoch {epoch} "
            f"for model {model_number}."
        )

    layer_names = sorted(
        df["layer"]
        .dropna()
        .astype(str)
        .unique()
    )

    # Do not create a histogram for the synthetic/global diagnostic layer.
    layer_names = [
        layer
        for layer in layer_names
        if layer.lower() not in {
            "global",
            "other"
        }
    ]

    # Gradient type in CSV -> requested output filename.
    # scaled_reg is the regularizer AFTER norm-ratio scaling.
    # combined is mse + scaled_reg.
    plot_definitions = {
        "mse": {"filename": "mse.png"},
        "scaled_reg": {"filename": "reg.png"},
        "combined": {"filename": "total.png"},
    }

    output_root.mkdir(
        parents=True,
        exist_ok=True
    )

    for layer_name in layer_names:

        layer_dir = (
            output_root
            / sanitize_name(layer_name)
        )

        layer_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        print(
            "\n"
            f"Layer: {layer_name}"
        )

        for loss_part, config in plot_definitions.items():

            selection = df[
                (df["layer"].astype(str) == layer_name)
                &
                (df["loss_part"].astype(str) == loss_part)
            ]

            values = pd.to_numeric(
                selection["gradient"],
                errors="coerce"
            ).dropna().to_numpy()

            output_path = (
                layer_dir
                / config["filename"]
            )

            save_gradient_histogram(
                values=values,
                layer_name=layer_name,
                loss_part=loss_part,
                output_path=output_path,
                bins=BINS,
                smooth_sigma=SMOOTH_SIGMA
            )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    script_dir = (
        Path(__file__)
        .resolve()
        .parent
    )

    print(
        f"Script directory: {script_dir}"
    )

    # Detect k_model_n.pth / K_model_n.pth automatically.
    _, model_number = detect_model(
        script_dir
    )

    # Detect latest gradient CSV produced for that model.
    gradient_csv, epoch = find_latest_gradient_csv(
        script_dir,
        model_number
    )

    df = pd.read_csv(
        gradient_csv
    )

    output_root = (
        script_dir
        / "gradients_histograms"
    )

    build_histograms(
        df=df,
        output_root=output_root,
        model_number=model_number,
        epoch=epoch
    )

    print(
        "\nHistogram generation completed."
    )
    print(
        f"Output folder: {output_root}"
    )


if __name__ == "__main__":
    main()
