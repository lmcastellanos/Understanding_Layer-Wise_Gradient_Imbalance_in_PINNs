#!/usr/bin/env python3
"""
Create gradient histograms for the 5-hidden-layer chemical-kinetics model.

This script reads:
    gradients_5_hidden_layers_model_<n>/gradient_values_by_layer.csv

produced by:
    calculate_gradients_5_hidden_layers.py

Network structure:
    architecture_e = [300, 300, 300, 300, 300]

Therefore the histogram routine expects:
    Layer 1
    Layer 2
    Layer 3
    Layer 4
    Layer 5
    Output layer

i.e. 6 Linear layers in total.

For every layer, two figures are generated:
    mse.png
    mass.png

The figures use the same publication-style format as the previous
gradient histograms:
    - 13 x 8 figure
    - no title
    - grid on
    - thick density curve
    - large axis labels/ticks
    - 600 dpi
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# USER SETTINGS
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent

EXPECTED_HIDDEN_LAYERS = 5
EXPECTED_TOTAL_LINEAR_LAYERS = 6

BINS = 60
SMOOTH_SIGMA = 2.0

FIGSIZE = (13, 8)
DPI = 600


# =============================================================================
# PLOT STYLE
# =============================================================================

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 30,
    "axes.labelsize": 30,
    "xtick.labelsize": 30,
    "ytick.labelsize": 30,
    "legend.fontsize": 24,
    "lines.linewidth": 3.0,
    "figure.dpi": 300,
    "savefig.dpi": DPI,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# =============================================================================
# DISCOVERY
# =============================================================================

def detect_gradient_folder(script_dir: Path):
    """
    Detect exactly one folder of the form:
        gradients_5_hidden_layers_model_<n>
    """

    pattern = re.compile(
        r"^gradients_5_hidden_layers_model_([0-4])$"
    )

    found = []

    for path in script_dir.iterdir():
        if not path.is_dir():
            continue

        match = pattern.match(path.name)
        if match:
            csv_path = (
                path
                / "gradient_values_by_layer.csv"
            )

            if csv_path.exists():
                found.append(
                    (
                        int(match.group(1)),
                        path,
                        csv_path,
                    )
                )

    if not found:
        raise FileNotFoundError(
            "\nNo gradient folder was found.\n\n"
            "Expected something like:\n"
            "    gradients_5_hidden_layers_model_0/\n"
            "        gradient_values_by_layer.csv\n"
        )

    if len(found) > 1:
        text = "\n".join(
            f"    model {idx}: {folder}"
            for idx, folder, _ in found
        )

        raise RuntimeError(
            "\nMore than one gradient folder was found.\n"
            "Keep only the case you want to plot in the script directory.\n\n"
            f"{text}"
        )

    return found[0]


# =============================================================================
# SMOOTHING
# =============================================================================

def gaussian_kernel(
    size: int = 13,
    sigma: float = 2.0,
):
    x = np.arange(size) - (
        size - 1
    ) / 2

    kernel = np.exp(
        -(x ** 2)
        / (2 * sigma ** 2)
    )

    kernel /= kernel.sum()

    return kernel


def smooth_density_from_histogram(
    values,
    bins=60,
    smooth_sigma=2.0,
):
    counts, bin_edges = np.histogram(
        values,
        bins=bins,
        density=True,
    )

    centers = 0.5 * (
        bin_edges[:-1]
        + bin_edges[1:]
    )

    kernel_size = max(
        7,
        int(6 * smooth_sigma) | 1,
    )

    kernel = gaussian_kernel(
        kernel_size,
        sigma=smooth_sigma,
    )

    smooth_counts = np.convolve(
        counts,
        kernel,
        mode="same",
    )

    return centers, smooth_counts


# =============================================================================
# LABELS / SORTING
# =============================================================================

def layer_sort_key(layer_label):
    """
    Order:
        Layer 1 ... Layer 5, Output layer
    """

    text = str(layer_label)

    match = re.match(
        r"Layer\s+(\d+)",
        text,
        flags=re.IGNORECASE,
    )

    if match:
        return (
            0,
            int(match.group(1)),
        )

    if "output" in text.lower():
        return (
            1,
            999,
        )

    return (
        2,
        text,
    )


def folder_name_from_layer(layer_label):
    text = str(layer_label)

    match = re.match(
        r"Layer\s+(\d+)",
        text,
        flags=re.IGNORECASE,
    )

    if match:
        return (
            f"layer_{int(match.group(1))}"
        )

    if "output" in text.lower():
        return "output_layer"

    return (
        text.lower()
        .replace(" ", "_")
    )


def loss_axis_label(loss_part):
    if loss_part == "mse":
        return r"$\mathcal{L}_{MSE}$"

    if loss_part == "mass":
        return r"$\mathcal{L}_{mass}$"

    return str(loss_part)


# =============================================================================
# HISTOGRAM
# =============================================================================

def save_histogram(
    values,
    loss_part,
    save_path,
):
    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[
        np.isfinite(values)
    ]

    if values.size == 0:
        print(
            f"Skipping empty gradient set: {save_path}"
        )
        return

    fig, ax = plt.subplots(
        figsize=FIGSIZE
    )

    # Histogram.
    ax.hist(
        values,
        bins=BINS,
        density=True,
        alpha=0.55,
        label="Gradient distribution",
    )

    # Smoothed density curve.
    if (
        values.size > 1
        and not np.allclose(
            values,
            values[0],
        )
    ):
        x_smooth, y_smooth = (
            smooth_density_from_histogram(
                values,
                bins=BINS,
                smooth_sigma=SMOOTH_SIGMA,
            )
        )

        ax.plot(
            x_smooth,
            y_smooth,
            linewidth=3.0,
            label="Smoothed density",
        )

    # Zero-gradient reference.
    ax.axvline(
        0.0,
        linestyle="--",
        linewidth=1.5,
    )

    # No title, for consistency with the previous figures.
    ax.set_xlabel(
        f"Gradient value of {loss_axis_label(loss_part)}"
    )

    ax.set_ylabel(
        "Density"
    )

    ax.grid(
        True,
        which="both",
        linestyle="--",
        linewidth=0.8,
        alpha=0.45,
    )

    ax.legend(
        frameon=False,
    )

    fig.tight_layout()

    fig.savefig(
        save_path,
        dpi=DPI,
        bbox_inches="tight",
    )

    plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================

def main():
    (
        model_index,
        gradient_folder,
        csv_path,
    ) = detect_gradient_folder(
        SCRIPT_DIR
    )

    print("=" * 78)
    print("5-HIDDEN-LAYER GRADIENT HISTOGRAMS")
    print("=" * 78)
    print(
        f"Model index : {model_index}"
    )
    print(
        f"Input CSV   : {csv_path}"
    )

    df = pd.read_csv(
        csv_path
    )

    required_columns = {
        "layer_index",
        "layer_label",
        "loss_part",
        "gradient",
    }

    missing = (
        required_columns
        - set(df.columns)
    )

    if missing:
        raise KeyError(
            "\nThe gradient CSV is missing required columns:\n"
            f"    {sorted(missing)}\n\n"
            f"Available columns:\n"
            f"    {list(df.columns)}"
        )

    df["gradient"] = pd.to_numeric(
        df["gradient"],
        errors="coerce",
    )

    df = df.dropna(
        subset=[
            "layer_label",
            "loss_part",
            "gradient",
        ]
    )

    layers = sorted(
        df["layer_label"]
        .astype(str)
        .unique()
        .tolist(),
        key=layer_sort_key,
    )

    print(
        f"Detected layers: {len(layers)}"
    )

    for layer in layers:
        print(
            f"    {layer}"
        )

    if (
        len(layers)
        != EXPECTED_TOTAL_LINEAR_LAYERS
    ):
        raise RuntimeError(
            "\nUnexpected number of layers in the gradient CSV.\n"
            f"Expected {EXPECTED_TOTAL_LINEAR_LAYERS} "
            f"({EXPECTED_HIDDEN_LAYERS} hidden + output), "
            f"but found {len(layers)}.\n"
        )

    output_root = (
        gradient_folder
        / "gradients_histograms"
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    for layer in layers:
        layer_df = df[
            df["layer_label"].astype(str)
            == str(layer)
        ]

        layer_dir = (
            output_root
            / folder_name_from_layer(
                layer
            )
        )

        layer_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        for loss_part in [
            "mse",
            "mass",
        ]:
            values = layer_df.loc[
                layer_df["loss_part"]
                == loss_part,
                "gradient",
            ].to_numpy()

            save_path = (
                layer_dir
                / f"{loss_part}.png"
            )

            save_histogram(
                values,
                loss_part,
                save_path,
            )

            print(
                f"Saved: {save_path}"
            )

    print("\nDone.")
    print(
        f"Histogram root:\n    {output_root}"
    )


if __name__ == "__main__":
    main()
