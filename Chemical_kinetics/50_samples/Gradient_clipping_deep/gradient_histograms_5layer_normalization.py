#!/usr/bin/env python3
"""
Five-layer gradient histogram analysis for chemical-kinetics models trained
with one-sided gradient clipping / gradient normalization.

What this script does
---------------------
1. Detects exactly one final model:
       k_model_0.pth ... k_model_4.pth
   or the uppercase-K equivalents.

2. Uses the detected model number as the cross-validation index.

3. Reconstructs Datasets_50 exactly as in the chemical-kinetics training:
       T + 8 species
       log(data + 1e-8)
       global normalization
       one-step shift
       RepeatedKFold(n_splits=5, random_state=42)

4. Recomputes independent gradients of:
       L_MSE
       L_mass

5. Finds ALL nn.Linear layers automatically. The intended network has
   5 Linear layers, and the script checks this explicitly.

6. Applies the one-sided gradient-normalization rule independently by layer:

       s_l = min(1, ||g_MSE^(l)|| / (||g_mass^(l)|| + eps))

       g_mass,scaled^(l) = s_l * g_mass^(l)

   Thus the plotted L_mass histograms correspond to the REGULARIZED /
   NORMALIZED mass-gradient contribution, not the raw mass gradients.

7. Saves, for each layer:
       mse.png
       mass_scaled.png

   It also saves:
       gradient_values.csv
       layer_gradient_summary.csv

The histograms have no titles, use a grid, and follow the same large-label
publication style used in the recent figures.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.model_selection import RepeatedKFold


# =============================================================================
# USER SETTINGS
# =============================================================================

DATASET_ROOT = Path.home() / "Alternative_trainings" / "Datasets_50"

# By default, search for k_model_n.pth in the same folder as this script.
MODEL_DIR = Path(__file__).resolve().parent

# Expected number of Linear layers in this case.
EXPECTED_LINEAR_LAYERS = 5

EPS = 1e-12
BINS = 60
SMOOTH_SIGMA = 2.0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# None = use the full training fold.
# Set an integer only for a quicker diagnostic.
MAX_TRAIN_SAMPLES = None
SUBSAMPLE_SEED = 42


# =============================================================================
# PLOT STYLE
# =============================================================================

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 30,
    "axes.labelsize": 30,
    "xtick.labelsize": 30,
    "ytick.labelsize": 30,
    "legend.fontsize": 26,
    "figure.dpi": 300,
    "savefig.dpi": 600,
    "lines.linewidth": 3.0,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# =============================================================================
# MODEL CLASS
# =============================================================================
# This generic class supports any architecture_e length, including a
# five-Linear-layer network saved with torch.save(model, ...).

class NN(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        dropout_p,
        architecture_e,
        min_val=None,
        max_val=None,
        alpha=None,
    ):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.architecture_e = architecture_e
        self.min_val = min_val
        self.max_val = max_val
        self.alpha = alpha

        encoder = []

        if len(architecture_e) == 1:
            layer = nn.Linear(input_size, architecture_e[0])
            nn.init.xavier_normal_(layer.weight)
            encoder.append(layer)
            encoder.append(nn.Sigmoid())
            encoder.append(nn.Dropout(p=dropout_p))

            layer = nn.Linear(architecture_e[0], output_size)
            nn.init.xavier_normal_(layer.weight)
            encoder.append(layer)

        else:
            for i in range(len(architecture_e)):
                if i == 0:
                    layer = nn.Linear(input_size, architecture_e[i])
                else:
                    layer = nn.Linear(architecture_e[i - 1], architecture_e[i])

                nn.init.xavier_normal_(layer.weight)
                encoder.append(layer)
                encoder.append(nn.Sigmoid())
                encoder.append(nn.Dropout(p=dropout_p))

            encoder.append(nn.Linear(architecture_e[-1], output_size))

        self.encoder = nn.Sequential(*encoder)

    def forward(self, x):
        return self.encoder(x)


# =============================================================================
# MODEL DETECTION
# =============================================================================

def detect_model(model_dir: Path):
    """
    Detect exactly one k_model_n.pth / K_model_n.pth and use n as MODEL_INDEX.
    """
    pattern = re.compile(r"^[kK]_model_([0-4])\.pth$")

    detected = []

    for path in model_dir.iterdir():
        if not path.is_file():
            continue

        match = pattern.match(path.name)
        if match:
            detected.append((int(match.group(1)), path))

    if not detected:
        raise FileNotFoundError(
            "\nNo final model found.\n"
            f"Searched in:\n{model_dir}\n\n"
            "Expected exactly one file named:\n"
            "k_model_0.pth ... k_model_4.pth"
        )

    if len(detected) > 1:
        names = ", ".join(path.name for _, path in detected)
        raise RuntimeError(
            "More than one final model was detected. "
            "Keep only the model you want to analyze in this folder.\n"
            f"Found: {names}"
        )

    model_index, model_path = detected[0]
    return model_index, model_path


# =============================================================================
# DATASET RECONSTRUCTION
# =============================================================================

def reconstruct_dataset_50(dataset_root: Path):
    sampling_file = dataset_root / "T_Sampling_uniform_50.csv"

    if not sampling_file.exists():
        raise FileNotFoundError(f"Missing sampling file:\n{sampling_file}")

    sampling = pd.read_csv(sampling_file)

    if "T[K]" in sampling.columns:
        temperatures = sampling["T[K]"].to_numpy()
    else:
        temperatures = sampling.iloc[:, 0].to_numpy()

    trajectories = []

    for T in temperatures:
        filename = (
            f"State_space_cte_pressure_T{round(float(T), 3)}_phi_1.0.csv"
        )
        path = dataset_root / filename

        if not path.exists():
            raise FileNotFoundError(f"Missing trajectory:\n{path}")

        df = pd.read_csv(path)

        temperature = df.iloc[:, 2].to_numpy().reshape(-1, 1)
        species = df.iloc[:, 4:12].to_numpy()

        data = np.concatenate([temperature, species], axis=1)
        data_log = np.log(data + 1e-8)

        trajectories.append(data_log)

    dataset = np.concatenate(trajectories, axis=0)

    means = dataset.mean(axis=0)
    stds = dataset.std(axis=0)
    stds = np.where(stds == 0.0, 1.0, stds)

    normalized = (dataset - means) / stds

    inputs = normalized[:-1, :]
    outputs = normalized[1:, :]

    return inputs, outputs, means, stds


def select_training_fold(inputs, outputs, model_index):
    splitter = RepeatedKFold(
        n_splits=5,
        n_repeats=1,
        random_state=42,
    )

    chosen = None

    for fold_idx, (train_index, _) in enumerate(splitter.split(inputs)):
        if fold_idx == model_index:
            chosen = (
                inputs[train_index],
                outputs[train_index],
            )
            break

    if chosen is None:
        raise ValueError(f"Invalid model index: {model_index}")

    x_train, y_train = chosen

    if MAX_TRAIN_SAMPLES is not None and len(x_train) > MAX_TRAIN_SAMPLES:
        rng = np.random.default_rng(SUBSAMPLE_SEED)
        idx = rng.choice(
            len(x_train),
            size=MAX_TRAIN_SAMPLES,
            replace=False,
        )
        x_train = x_train[idx]
        y_train = y_train[idx]

    return (
        x_train.astype("float32"),
        y_train.astype("float32"),
    )


# =============================================================================
# MODEL LOADING
# =============================================================================

def load_model(model_path: Path):
    try:
        model = torch.load(
            model_path,
            map_location=DEVICE,
            weights_only=False,
        )
    except TypeError:
        model = torch.load(
            model_path,
            map_location=DEVICE,
        )

    if isinstance(model, nn.DataParallel):
        model = model.module

    if not isinstance(model, nn.Module):
        raise TypeError(
            f"{model_path} does not contain a complete torch.nn.Module."
        )

    model = model.to(DEVICE)
    model.eval()

    return model


def infer_model_dtype(model):
    for p in model.parameters():
        return p.dtype
    return torch.float32


# =============================================================================
# LOSS FUNCTIONS
# =============================================================================

def compute_losses(model, x, y, means_t, stds_t):
    y_pred = model(x)

    # Data-fitting loss.
    mse = torch.mean(torch.square(y - y_pred))

    # Mass conservation uses only species columns, excluding temperature.
    pred_rescaled = (
        y_pred[:, 1:] * stds_t[1:]
        + means_t[1:]
    )

    true_rescaled = (
        y[:, 1:] * stds_t[1:]
        + means_t[1:]
    )

    pred_species = torch.exp(pred_rescaled)
    true_species = torch.exp(true_rescaled)

    mass_loss = torch.square(
        torch.sum(true_species, dim=1)
        - torch.sum(pred_species, dim=1)
    ).mean()

    return mse.float(), mass_loss.float()


# =============================================================================
# GRADIENT UTILITIES
# =============================================================================

def separate_gradients(loss, params, retain_graph):
    return torch.autograd.grad(
        loss,
        params,
        retain_graph=retain_graph,
        create_graph=False,
        allow_unused=True,
    )


def get_linear_layer_groups(model, params):
    """
    Return parameter indices belonging to every nn.Linear layer.

    Weight and bias are grouped together to define each layer's scaling factor,
    but their individual gradient ELEMENTS are preserved for the histograms.
    """
    param_id_to_index = {
        id(p): i for i, p in enumerate(params)
    }

    groups = []
    layer_counter = 0

    for module_name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue

        layer_counter += 1
        indices = []

        if module.weight is not None:
            idx = param_id_to_index.get(id(module.weight))
            if idx is not None:
                indices.append(idx)

        if module.bias is not None:
            idx = param_id_to_index.get(id(module.bias))
            if idx is not None:
                indices.append(idx)

        groups.append({
            "layer_index": layer_counter,
            "module_name": module_name,
            "param_indices": indices,
        })

    return groups


def joint_gradient_norm(grads, indices):
    total_sq = 0.0

    for i in indices:
        g = grads[i]

        if g is None:
            continue

        total_sq += float(
            torch.sum(g.detach().double() ** 2)
            .cpu()
            .item()
        )

    return float(np.sqrt(total_sq))


def flatten_layer_gradients(grads, indices, scale=1.0):
    """
    Flatten all weight+bias gradient elements from one layer.
    """
    arrays = []

    for i in indices:
        g = grads[i]

        if g is None:
            continue

        values = (
            scale
            * g.detach()
            .reshape(-1)
            .cpu()
            .numpy()
        )

        arrays.append(values)

    if not arrays:
        return np.array([], dtype=float)

    return np.concatenate(arrays).astype(float)


# =============================================================================
# HISTOGRAM UTILITIES
# =============================================================================

def gaussian_kernel(size=13, sigma=2.0):
    x = np.arange(size) - (size - 1) / 2
    kernel = np.exp(-(x ** 2) / (2 * sigma ** 2))
    kernel /= kernel.sum()
    return kernel


def smooth_density_from_histogram(values, bins=60, smooth_sigma=2.0):
    counts, edges = np.histogram(
        values,
        bins=bins,
        density=True,
    )

    centers = 0.5 * (edges[:-1] + edges[1:])

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


def save_histogram(values, xlabel, save_path):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if values.size == 0:
        print(f"WARNING: no finite gradients for {save_path}")
        return

    fig, ax = plt.subplots(
        figsize=(13, 8),
        dpi=300,
    )

    ax.hist(
        values,
        bins=BINS,
        density=True,
        alpha=0.55,
    )

    if values.size > 1 and not np.allclose(values, values[0]):
        x_smooth, y_smooth = smooth_density_from_histogram(
            values,
            bins=BINS,
            smooth_sigma=SMOOTH_SIGMA,
        )

        ax.plot(
            x_smooth,
            y_smooth,
            linewidth=3.0,
        )

    # No title: same style as the recent publication figures.
    ax.set_xlabel(xlabel, fontsize=30)
    ax.set_ylabel("Density", fontsize=30)

    ax.tick_params(
        axis="both",
        which="major",
        labelsize=30,
    )

    ax.grid(
        True,
        alpha=0.30,
        linestyle="--",
        linewidth=1.2,
    )

    fig.tight_layout()

    fig.savefig(
        save_path,
        dpi=600,
        bbox_inches="tight",
    )

    plt.close(fig)


# =============================================================================
# ANALYSIS
# =============================================================================

def main():
    model_index, model_path = detect_model(MODEL_DIR)

    print("=" * 78)
    print("FIVE-LAYER GRADIENT HISTOGRAM ANALYSIS")
    print("=" * 78)
    print(f"Detected model : {model_path.name}")
    print(f"Model index    : {model_index}")
    print(f"Device         : {DEVICE}")
    print()

    inputs, outputs, means, stds = reconstruct_dataset_50(
        DATASET_ROOT
    )

    x_train, y_train = select_training_fold(
        inputs,
        outputs,
        model_index,
    )

    print(f"Training samples used: {len(x_train)}")

    model = load_model(model_path)
    dtype = infer_model_dtype(model)

    x = torch.as_tensor(
        x_train,
        dtype=dtype,
        device=DEVICE,
    )

    y = torch.as_tensor(
        y_train,
        dtype=dtype,
        device=DEVICE,
    )

    means_t = torch.as_tensor(
        means,
        dtype=dtype,
        device=DEVICE,
    )

    stds_t = torch.as_tensor(
        stds,
        dtype=dtype,
        device=DEVICE,
    )

    mse_loss, mass_loss = compute_losses(
        model,
        x,
        y,
        means_t,
        stds_t,
    )

    params = [
        p for p in model.parameters()
        if p.requires_grad
    ]

    mse_grads = separate_gradients(
        mse_loss,
        params,
        retain_graph=True,
    )

    mass_grads = separate_gradients(
        mass_loss,
        params,
        retain_graph=False,
    )

    layer_groups = get_linear_layer_groups(
        model,
        params,
    )

    print(f"Linear layers detected: {len(layer_groups)}")

    if len(layer_groups) != EXPECTED_LINEAR_LAYERS:
        details = "\n".join(
            f"  Layer {g['layer_index']}: {g['module_name']}"
            for g in layer_groups
        )
        raise RuntimeError(
            f"\nExpected {EXPECTED_LINEAR_LAYERS} Linear layers, "
            f"but detected {len(layer_groups)}.\n"
            f"Detected layers:\n{details}"
        )

    output_root = (
        MODEL_DIR
        / f"gradients_histograms_model_{model_index}_5layers"
    )
    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    value_rows = []
    summary_rows = []

    print()
    print(f"L_MSE  = {float(mse_loss.detach().cpu()):.6e}")
    print(f"L_mass = {float(mass_loss.detach().cpu()):.6e}")
    print()

    for group in layer_groups:
        layer_index = group["layer_index"]
        indices = group["param_indices"]

        mse_norm = joint_gradient_norm(
            mse_grads,
            indices,
        )

        mass_norm_raw = joint_gradient_norm(
            mass_grads,
            indices,
        )

        # -------------------------------------------------------------
        # One-sided layer-wise gradient normalization / clipping.
        # -------------------------------------------------------------
        if mass_norm_raw > mse_norm:
            scale = mse_norm / (mass_norm_raw + EPS)
        else:
            scale = 1.0

        mass_norm_scaled = scale * mass_norm_raw

        mse_values = flatten_layer_gradients(
            mse_grads,
            indices,
            scale=1.0,
        )

        mass_values_raw = flatten_layer_gradients(
            mass_grads,
            indices,
            scale=1.0,
        )

        mass_values_scaled = flatten_layer_gradients(
            mass_grads,
            indices,
            scale=scale,
        )

        layer_dir = output_root / f"layer_{layer_index}"
        layer_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        save_histogram(
            mse_values,
            xlabel=r"Gradient value of $\mathcal{L}_{MSE}$",
            save_path=layer_dir / "mse.png",
        )

        save_histogram(
            mass_values_scaled,
            xlabel=r"Gradient value of scaled $\mathcal{L}_{mass}$",
            save_path=layer_dir / "mass_scaled.png",
        )

        # Long-form gradient-element CSV.
        for value in mse_values:
            value_rows.append({
                "model_index": model_index,
                "layer_index": layer_index,
                "loss_part": "mse",
                "gradient": value,
            })

        for value in mass_values_scaled:
            value_rows.append({
                "model_index": model_index,
                "layer_index": layer_index,
                "loss_part": "mass_scaled",
                "gradient": value,
            })

        summary_rows.append({
            "model_index": model_index,
            "layer_index": layer_index,
            "module_name": group["module_name"],
            "mse_loss": float(mse_loss.detach().cpu()),
            "mass_loss": float(mass_loss.detach().cpu()),
            "mse_grad_norm": mse_norm,
            "mass_grad_norm_raw": mass_norm_raw,
            "normalization_scale": scale,
            "mass_grad_norm_scaled": mass_norm_scaled,
            "raw_R_l": mass_norm_raw / (mse_norm + EPS),
            "scaled_R_l": mass_norm_scaled / (mse_norm + EPS),
            "n_mse_gradient_elements": len(mse_values),
            "n_mass_gradient_elements": len(mass_values_scaled),
        })

        print(
            f"Layer {layer_index}: "
            f"||g_MSE||={mse_norm:.4e}, "
            f"||g_mass_raw||={mass_norm_raw:.4e}, "
            f"scale={scale:.4e}, "
            f"||g_mass_scaled||={mass_norm_scaled:.4e}"
        )

    gradient_values_df = pd.DataFrame(value_rows)
    summary_df = pd.DataFrame(summary_rows)

    gradient_csv = output_root / "gradient_values.csv"
    summary_csv = output_root / "layer_gradient_summary.csv"

    gradient_values_df.to_csv(
        gradient_csv,
        index=False,
    )

    summary_df.to_csv(
        summary_csv,
        index=False,
    )

    print()
    print("=" * 78)
    print("DONE")
    print("=" * 78)
    print(f"Output folder : {output_root}")
    print(f"Gradient CSV  : {gradient_csv}")
    print(f"Summary CSV   : {summary_csv}")
    print()
    print("Per-layer figures:")
    print("  layer_1/mse.png")
    print("  layer_1/mass_scaled.png")
    print("  ...")
    print("  layer_5/mse.png")
    print("  layer_5/mass_scaled.png")


if __name__ == "__main__":
    main()
