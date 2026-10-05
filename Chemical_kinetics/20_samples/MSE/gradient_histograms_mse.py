#Library

#gradient_histograms_mse_auto_model.py


#!/usr/bin/env python3

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# USER SETTINGS
# ============================================================

BINS = 60
SMOOTH_SIGMA = 2.0

REQUIRED_COLUMNS = {
    "run_id",
    "repeat_folder",
    "epoch",
    "loss_part",
    "layer",
    "gradient",
}


# Publication-style formatting
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


# ============================================================
# GENERAL UTILITIES
# ============================================================

def sanitize_name(name: str) -> str:
    return (
        str(name)
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
    )


def gaussian_kernel(size: int = 13, sigma: float = 2.0) -> np.ndarray:
    x = np.arange(size) - (size - 1) / 2
    kernel = np.exp(-(x ** 2) / (2 * sigma ** 2))
    kernel /= kernel.sum()
    return kernel


def smooth_density_from_histogram(
    values: np.ndarray,
    bins: int = 60,
    smooth_sigma: float = 2.0,
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


# ============================================================
# AUTOMATIC MODEL DETECTION
# ============================================================

def detect_model(script_dir: Path):
    """
    Recursively detect exactly one final MSE model:

        k_model_0.pth ... k_model_4.pth
        K_model_0.pth ... K_model_4.pth

    Returns
    -------
    model_number : int
    model_path : Path
    repeat_folder : str
    """

    pattern = re.compile(
        r"^[Kk]_model_([0-4])\.pth$"
    )

    detected = []

    for path in sorted(
        script_dir.rglob("*.pth")
    ):
        match = pattern.match(
            path.name
        )

        if match is None:
            continue

        detected.append(
            (
                int(match.group(1)),
                path,
            )
        )

    if len(detected) == 0:
        raise FileNotFoundError(
            "No final MSE model was found. Expected exactly one "
            "k_model_n.pth or K_model_n.pth with n between 0 and 4."
        )

    if len(detected) > 1:
        names = ", ".join(
            str(path)
            for _, path in detected
        )

        raise RuntimeError(
            "More than one final MSE model was detected. "
            f"Found: {names}"
        )

    model_number, model_path = detected[0]
    repeat_folder = model_path.parent.name

    print("Detected model:")
    print(f"  file          : {model_path}")
    print(f"  model number  : {model_number}")
    print(f"  repeat folder : {repeat_folder}")

    return (
        model_number,
        model_path,
        repeat_folder,
    )


# ============================================================
# FIND MSE GRADIENT CSV
# ============================================================

def find_latest_mse_gradient_csv(
    script_dir: Path,
    model_number: int,
    repeat_folder: str,
):
    """
    Expected output from the MSE-only gradient calculator:

        gradients_by_run_mse/
            repeats_1/
                run_n/
                    gradients_run_n_epoch_x.csv

    The latest epoch is selected automatically.
    """

    root = (
        script_dir
        / "gradients_by_run_mse"
    )

    if not root.exists():
        raise FileNotFoundError(
            f"MSE gradient root not found: {root}"
        )

    preferred_dir = (
        root
        / repeat_folder
        / f"run_{model_number}"
    )

    if preferred_dir.exists():
        candidates = sorted(
            preferred_dir.rglob("*.csv")
        )
    else:
        # Fallback recursive search by run ID.
        candidates = []

        pattern = re.compile(
            rf"gradients_run_{model_number}_epoch_(\d+)\.csv$"
        )

        for path in sorted(
            root.rglob("*.csv")
        ):
            if pattern.match(path.name):
                candidates.append(path)

    usable = []

    filename_pattern = re.compile(
        rf"gradients_run_{model_number}_epoch_(\d+)\.csv$"
    )

    for csv_path in candidates:

        match = filename_pattern.match(
            csv_path.name
        )

        if match is None:
            continue

        try:
            preview = pd.read_csv(
                csv_path,
                nrows=5,
            )
        except Exception:
            continue

        if not REQUIRED_COLUMNS.issubset(
            preview.columns
        ):
            continue

        epoch = int(
            match.group(1)
        )

        usable.append(
            (
                epoch,
                csv_path,
            )
        )

    if len(usable) == 0:
        raise FileNotFoundError(
            f"No usable MSE-gradient CSV was found "
            f"for model {model_number}."
        )

    usable.sort(
        key=lambda item: item[0]
    )

    epoch, csv_path = usable[-1]

    print(
        f"Using gradient CSV: {csv_path}"
    )
    print(
        f"Latest detected epoch: {epoch}"
    )

    return csv_path, epoch


# ============================================================
# HISTOGRAM CREATION
# ============================================================

def save_gradient_histogram(
    values,
    layer_name,
    save_path: Path,
    epoch=None,
    bins=60,
    smooth_sigma=2.0,
):
    """
    Publication-style MSE-gradient histogram with:
      - normalized histogram bars
      - smoothed density curve
      - compact N / mean / std statistics box
    """

    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[
        np.isfinite(values)
    ]

    if values.size == 0:
        return False

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    # Histogram bars
    ax.hist(
        values,
        bins=bins,
        density=True,
        alpha=0.55,
        label="Gradient distribution",
    )

    # Smoothed density
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
                bins=bins,
                smooth_sigma=smooth_sigma,
            )
        )

        ax.plot(
            x_smooth,
            y_smooth,
            linewidth=2.0,
            label="Smoothed density",
        )

    title = (
        f"Layer: {layer_name} | "
        r"Gradients of $\mathcal{L}_{MSE}$"
    )

    if epoch is not None:
        title += (
            f" | Epoch {epoch}"
        )

    ax.set_title(
        title
    )

    ax.set_xlabel(
        "Gradient value"
    )

    ax.set_ylabel(
        "Density"
    )

    ax.legend()

    # Compact statistics box
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
        bbox=dict(
            boxstyle="round",
            alpha=0.15,
        ),
    )

    fig.tight_layout()

    fig.savefig(
        save_path,
        dpi=600,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Saved: {save_path}"
    )

    return True


# ============================================================
# BUILD HISTOGRAMS PER LAYER
# ============================================================

def build_histograms(
    df: pd.DataFrame,
    output_root: Path,
    model_number: int,
    epoch: int,
):
    """
    Output:

        gradients_histograms_mse/
            hidden_1/
                mse.png
            hidden_2/
                mse.png
            hidden_3/
                mse.png
            output/
                mse.png
    """

    missing = (
        REQUIRED_COLUMNS
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            "Gradient CSV is missing required columns: "
            f"{sorted(missing)}"
        )

    df = df.copy()

    # Restrict to automatically detected model.
    df = df[
        pd.to_numeric(
            df["run_id"],
            errors="coerce",
        )
        == model_number
    ]

    # Restrict to selected epoch.
    df = df[
        pd.to_numeric(
            df["epoch"],
            errors="coerce",
        )
        == epoch
    ]

    # MSE only.
    df["loss_part"] = (
        df["loss_part"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    df = df[
        df["loss_part"] == "mse"
    ]

    if df.empty:
        raise ValueError(
            "No MSE-gradient rows were found "
            f"for model {model_number} at epoch {epoch}."
        )

    layer_names = sorted(
        df["layer"]
        .dropna()
        .astype(str)
        .unique()
    )

    # Skip non-network diagnostic groups.
    layer_names = [
        layer
        for layer in layer_names
        if layer.lower()
        not in {
            "global",
            "other",
        }
    ]

    if len(layer_names) == 0:
        raise ValueError(
            "No valid network layer names "
            "were found in the MSE gradient data."
        )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    saved_count = 0

    for layer_name in layer_names:

        layer_dir = (
            output_root
            / sanitize_name(
                layer_name
            )
        )

        layer_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        df_layer = df[
            df["layer"].astype(str)
            == layer_name
        ]

        values = pd.to_numeric(
            df_layer["gradient"],
            errors="coerce",
        ).dropna().to_numpy()

        save_path = (
            layer_dir
            / "mse.png"
        )

        saved = (
            save_gradient_histogram(
                values=values,
                layer_name=layer_name,
                save_path=save_path,
                epoch=epoch,
                bins=BINS,
                smooth_sigma=SMOOTH_SIGMA,
            )
        )

        if saved:
            saved_count += 1

    return saved_count


# ============================================================
# MAIN
# ============================================================

def main():

    script_dir = (
        Path(__file__)
        .resolve()
        .parent
    )

    (
        model_number,
        _model_path,
        repeat_folder,
    ) = detect_model(
        script_dir
    )

    gradient_csv, epoch = (
        find_latest_mse_gradient_csv(
            script_dir=script_dir,
            model_number=model_number,
            repeat_folder=repeat_folder,
        )
    )

    df = pd.read_csv(
        gradient_csv
    )

    output_root = (
        script_dir
        / "gradients_histograms_mse"
    )

    saved_count = (
        build_histograms(
            df=df,
            output_root=output_root,
            model_number=model_number,
            epoch=epoch,
        )
    )

    print(
        "\nMSE histogram generation complete."
    )

    print(
        f"Output folder: {output_root}"
    )

    print(
        f"Histograms saved: {saved_count}"
    )


if __name__ == "__main__":
    main()