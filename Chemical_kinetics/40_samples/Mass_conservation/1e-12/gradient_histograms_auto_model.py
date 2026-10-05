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
REQUIRED_COLUMNS = {"loss_part", "gradient", "layer"}
LOSS_ORDER = ["mse", "reg", "total"]

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
    """Create a filesystem-safe name from a layer name."""
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
    """Return histogram-bin centers and a smoothed density curve."""
    counts, bin_edges = np.histogram(values, bins=bins, density=True)
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    kernel_size = max(7, int(6 * smooth_sigma) | 1)
    kernel = gaussian_kernel(kernel_size, sigma=smooth_sigma)
    smooth_counts = np.convolve(counts, kernel, mode="same")

    return centers, smooth_counts


# ============================================================
# MODEL DETECTION
# ============================================================
def detect_model(script_dir: Path):
    """
    Detect exactly one model named k_model_n.pth or K_model_n.pth,
    where n is between 0 and 4.
    """
    pattern = re.compile(r"^[kK]_model_([0-4])\.pth$")

    detected = []
    for path in script_dir.iterdir():
        if not path.is_file():
            continue

        match = pattern.match(path.name)
        if match:
            detected.append((int(match.group(1)), path))

    if len(detected) == 0:
        raise FileNotFoundError(
            "No model was found in the script folder. Expected exactly one of: "
            "k_model_0.pth, ..., k_model_4.pth (uppercase K is also accepted)."
        )

    if len(detected) > 1:
        names = ", ".join(path.name for _, path in detected)
        raise RuntimeError(
            "More than one model was found. This routine expects exactly one model. "
            f"Found: {names}"
        )

    model_number, model_path = detected[0]
    return model_number, model_path


# ============================================================
# GRADIENT CSV DETECTION
# ============================================================
def extract_epoch_from_path(path: Path):
    """
    Try to infer an epoch from file/folder names.

    Recognized examples include:
        epoch_1000
        epoch1000
        epochs_2000
        ..._1000_...
    """
    texts = [path.stem] + [parent.name for parent in path.parents]

    # Prefer explicit 'epoch' labels.
    for text in texts:
        match = re.search(r"epochs?[_-]?(\d+)", text, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))

    # Fallback: use reasonably large integers, which are likely epochs.
    for text in texts:
        numbers = [int(x) for x in re.findall(r"(\d+)", text)]
        plausible = [x for x in numbers if x >= 100]
        if plausible:
            return max(plausible)

    return None


def is_gradient_csv(csv_path: Path) -> bool:
    try:
        preview = pd.read_csv(csv_path, nrows=5)
    except Exception:
        return False

    return REQUIRED_COLUMNS.issubset(set(preview.columns))


def find_gradient_csvs_for_model(script_dir: Path, model_number: int):
    """
    Find gradient CSV files corresponding to the detected model.

    Preferred layout:
        gradients_by_run/run_n/...csv

    where n is the automatically detected model number.

    If that exact run folder is not present, the function searches recursively
    under gradients_by_run and keeps gradient CSVs whose path explicitly contains
    run_n, model_n, or k_model_n.
    """
    root_dir = script_dir / "gradients_by_run"

    if not root_dir.exists():
        raise FileNotFoundError(
            f"The gradients folder was not found: {root_dir}"
        )

    preferred_run_dir = root_dir / f"run_{model_number}"

    if preferred_run_dir.exists():
        candidates = sorted(preferred_run_dir.rglob("*.csv"))
    else:
        candidates = []
        model_patterns = [
            re.compile(rf"(^|[^0-9])run[_-]?{model_number}([^0-9]|$)", re.IGNORECASE),
            re.compile(rf"(^|[^0-9])model[_-]?{model_number}([^0-9]|$)", re.IGNORECASE),
            re.compile(rf"[kK]_model_{model_number}", re.IGNORECASE),
        ]

        for csv_path in sorted(root_dir.rglob("*.csv")):
            path_text = str(csv_path)
            if any(pattern.search(path_text) for pattern in model_patterns):
                candidates.append(csv_path)

    usable = [path for path in candidates if is_gradient_csv(path)]

    if not usable:
        raise FileNotFoundError(
            f"No usable gradient CSV files were found for model {model_number}. "
            "Each CSV must contain the columns: loss_part, gradient, layer."
        )

    return usable


def choose_latest_gradient_snapshot(csv_paths):
    """
    Select CSV file(s) belonging to the latest detectable epoch.

    If epochs cannot be inferred from any path, all usable CSV files are used.
    Multiple CSV files at the same latest epoch are concatenated.
    """
    epoch_map = []
    for path in csv_paths:
        epoch_map.append((path, extract_epoch_from_path(path)))

    detected_epochs = [epoch for _, epoch in epoch_map if epoch is not None]

    if not detected_epochs:
        return csv_paths, None

    latest_epoch = max(detected_epochs)
    selected = [path for path, epoch in epoch_map if epoch == latest_epoch]

    return selected, latest_epoch


def load_and_combine_gradient_csvs(csv_paths):
    frames = []

    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)

        if not REQUIRED_COLUMNS.issubset(df.columns):
            continue

        df = df.copy()
        df["source_file"] = csv_path.name
        frames.append(df)

    if not frames:
        raise ValueError("No valid gradient data could be loaded.")

    return pd.concat(frames, ignore_index=True)


# ============================================================
# HISTOGRAM CREATION
# ============================================================
def save_gradient_histogram(
    values,
    layer_name,
    loss_part,
    save_path,
    model_number,
    epoch=None,
    bins=60,
    smooth_sigma=2.0,
):
    """
    Save one histogram. A smoothed density line is drawn over the histogram,
    while the output remains a single figure/file.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if values.size == 0:
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

    # Smoothed histogram density
    # Skip only in the pathological case where all values are identical.
    if values.size > 1 and not np.allclose(values, values[0]):
        x_smooth, y_smooth = smooth_density_from_histogram(
            values,
            bins=bins,
            smooth_sigma=smooth_sigma,
        )
        ax.plot(x_smooth, y_smooth, linewidth=2.0, label="Smoothed density")

    loss_label = {
        "mse": r"$\mathcal{L}_{MSE}$",
        "reg": r"$\mathcal{L}_{mass}$",
        "total": r"$\mathcal{L}_{total}$",
    }.get(loss_part, loss_part)

    title = f"Layer: {layer_name} | Gradients of {loss_label}"
    if epoch is not None:
        title += f" | Epoch {epoch}"

    ax.set_title(title)
    ax.set_xlabel("Gradient value")
    ax.set_ylabel("Density")
    ax.legend()

    # Add a compact statistics box.
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
    fig.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.close(fig)

    return True


def build_histograms(df, output_root: Path, model_number: int, epoch=None):
    """
    Required output structure:

        gradients_histograms/
            layer_1/
                mse.png
                reg.png
                total.png
            layer_2/
                mse.png
                reg.png
                total.png
            ...
    """
    output_root.mkdir(parents=True, exist_ok=True)

    # Normalize loss labels so variants such as MSE / REG / Total work.
    df = df.copy()
    df["loss_part"] = df["loss_part"].astype(str).str.strip().str.lower()

    aliases = {
        "mse_loss": "mse",
        "loss_mse": "mse",
        "regularization": "reg",
        "regularizer": "reg",
        "reg_loss": "reg",
        "loss_reg": "reg",
        "combined": "total",
        "combined_loss": "total",
        "total_loss": "total",
        "loss_total": "total",
    }
    df["loss_part"] = df["loss_part"].replace(aliases)

    layer_names = sorted(df["layer"].dropna().astype(str).unique())

    if len(layer_names) == 0:
        raise ValueError("No layer names were found in the gradient data.")

    saved_count = 0
    missing_combinations = []

    for layer_name in layer_names:
        layer_dir = output_root / sanitize_name(layer_name)
        layer_dir.mkdir(parents=True, exist_ok=True)

        df_layer = df[df["layer"].astype(str) == layer_name]

        for loss_part in LOSS_ORDER:
            df_loss = df_layer[df_layer["loss_part"] == loss_part]
            values = pd.to_numeric(df_loss["gradient"], errors="coerce").dropna().to_numpy()

            if values.size == 0:
                missing_combinations.append((layer_name, loss_part))
                continue

            save_path = layer_dir / f"{loss_part}.png"

            saved = save_gradient_histogram(
                values=values,
                layer_name=layer_name,
                loss_part=loss_part,
                save_path=save_path,
                model_number=model_number,
                epoch=epoch,
                bins=BINS,
                smooth_sigma=SMOOTH_SIGMA,
            )

            if saved:
                saved_count += 1

    return saved_count, missing_combinations


# ============================================================
# MAIN
# ============================================================
def main():
    script_dir = Path(__file__).resolve().parent

    # 1. Detect the only model automatically.
    model_number, model_path = detect_model(script_dir)

    print("Detected model:")
    print(f"  file   : {model_path.name}")
    print(f"  number : {model_number}")

    # 2. Find gradient CSVs corresponding to that model.
    gradient_csvs = find_gradient_csvs_for_model(
        script_dir=script_dir,
        model_number=model_number,
    )

    print(f"\nUsable gradient CSV files found: {len(gradient_csvs)}")
    for path in gradient_csvs:
        print(f"  {path}")

    # 3. Automatically select the latest available epoch.
    selected_csvs, latest_epoch = choose_latest_gradient_snapshot(gradient_csvs)

    if latest_epoch is not None:
        print(f"\nLatest detected epoch: {latest_epoch}")
        print("Using gradient file(s) from this epoch:")
    else:
        print("\nNo epoch could be inferred from the paths.")
        print("Using all matching gradient CSV files:")

    for path in selected_csvs:
        print(f"  {path}")

    # 4. Combine data if the latest snapshot is split across several CSVs.
    gradient_df = load_and_combine_gradient_csvs(selected_csvs)

    # 5. Produce the requested folder hierarchy.
    output_root = script_dir / "gradients_histograms"

    saved_count, missing = build_histograms(
        df=gradient_df,
        output_root=output_root,
        model_number=model_number,
        epoch=latest_epoch,
    )

    print("\nHistogram generation complete.")
    print(f"Output folder: {output_root}")
    print(f"Histograms saved: {saved_count}")

    if missing:
        print("\nWARNING: these layer/loss combinations were not present in the CSV data:")
        for layer_name, loss_part in missing:
            print(f"  layer={layer_name}, loss={loss_part}")
        print(
            "No histogram was generated for those missing combinations. "
            "In particular, a total-gradient histogram requires rows whose "
            "loss_part is 'total' (or an accepted alias such as 'total_loss')."
        )


if __name__ == "__main__":
    main()
