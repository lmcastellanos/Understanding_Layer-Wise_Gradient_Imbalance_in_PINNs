#!/usr/bin/env python3
"""
General H_R analysis for chemical-kinetics models with mass conservation.

This script is intended for BOTH:
    1) one-sided gradient clipping / gradient scaling models
    2) fixed-regularization models

It loads saved checkpoints, reconstructs the Datasets_50 training fold,
recomputes the MSE and raw mass-conservation losses, calculates separate
gradients, groups Linear-layer weight+bias tensors, and evaluates:

    R_l_raw = ||g_mass^(l)|| / (||g_MSE^(l)|| + eps)

    log10(R_l_raw)

    H_R = max_l log10(R_l_raw) - min_l log10(R_l_raw)

For fixed regularization, it also reports:

    R_l_beta = |beta| * R_l_raw

Note:
    H_R is invariant to a single global beta because the same log10(beta)
    shift is added to every layer.

For clipping/adaptive models, H_R is deliberately computed from the RAW
pre-clipping gradients, so it measures intrinsic layer heterogeneity.

---------------------------------------------------------------------------
EDIT ONLY THE USER SETTINGS SECTION BELOW.
---------------------------------------------------------------------------
"""

import os
import re
import sys
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

# "clipping" or "fixed"
MODE = "fixed"

# Only used for fixed regularization diagnostics.
# Examples: 1e-9, 1e-5, 1.0
BETA = 1e-9

# Which cross-validation/model index to analyze.
MODEL_INDEX = 3

# Dataset root.
DATASET_ROOT = Path.home() / "Alternative_trainings" / "Datasets_50"

# Folder containing the saved checkpoint .pth files.
# By default: the folder where this script is placed.
MODEL_DIR = Path(__file__).resolve().parent

# Optional recursive search for checkpoints below MODEL_DIR.
RECURSIVE_MODEL_SEARCH = True

# Numerical epsilon in R_l.
EPS = 1e-12

# Device.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set to an integer for a quicker diagnostic if full training data is too large.
# Keep None for the exact full fold.
MAX_TRAIN_SAMPLES = None

# Random seed used only if MAX_TRAIN_SAMPLES is not None.
SUBSAMPLE_SEED = 42


# =============================================================================
# MODEL CLASS
# =============================================================================
# Needed when checkpoints were saved with torch.save(model, ...).

class NN(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        dropout_p,
        architecture_e,
        min_val=None,
        max_val=None,
    ):
        super().__init__()

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
            encoder.append(nn.Sigmoid())
            encoder.append(nn.Dropout(p=dropout_p))

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
# DATASET RECONSTRUCTION
# =============================================================================

def reconstruct_dataset_50(dataset_root: Path):
    """
    Reproduces the Datasets_50 preprocessing used by the chemical-kinetics
    training scripts:

        - temperature column 2
        - species columns 4:12
        - log(data + 1e-8)
        - concatenate all trajectories
        - normalize over full concatenated dataset
        - one-step shift over the concatenated array
    """

    sampling_file = dataset_root / "T_Sampling_uniform_50.csv"
    if not sampling_file.exists():
        raise FileNotFoundError(
            f"Could not find sampling file:\n{sampling_file}"
        )

    sampling = pd.read_csv(sampling_file)

    # Try the usual column name first, otherwise use the first column.
    if "T[K]" in sampling.columns:
        temperatures = sampling["T[K]"].to_numpy()
    else:
        temperatures = sampling.iloc[:, 0].to_numpy()

    trajectories = []

    for T in temperatures:
        filename = (
            f"State_space_cte_pressure_T{round(float(T), 3)}_phi_1.0.csv"
        )
        file_path = dataset_root / filename

        if not file_path.exists():
            raise FileNotFoundError(
                f"Could not find state-space file:\n{file_path}"
            )

        df = pd.read_csv(file_path)

        # T + 8 species = 9 variables.
        temperature = df.iloc[:, 2].to_numpy().reshape(-1, 1)
        species = df.iloc[:, 4:12].to_numpy()

        data = np.concatenate(
            [temperature, species],
            axis=1,
        )

        data_log = np.log(data + 1e-8)
        trajectories.append(data_log)

    dataset = np.concatenate(
        trajectories,
        axis=0,
    )

    means = dataset.mean(axis=0)
    stds = dataset.std(axis=0)

    # Avoid accidental division by zero.
    stds = np.where(stds == 0.0, 1.0, stds)

    normalized = (dataset - means) / stds

    # Same one-step global shift used in the training source.
    inputs = normalized[:-1, :]
    outputs = normalized[1:, :]

    return inputs, outputs, means, stds


def select_training_fold(inputs, outputs, model_index):
    """
    Selects the same fold convention used in the training scripts:
        RepeatedKFold(n_splits=5, n_repeats=1, random_state=42)
    """

    splitter = RepeatedKFold(
        n_splits=5,
        n_repeats=1,
        random_state=42,
    )

    chosen = None

    for fold_idx, (train_index, test_index) in enumerate(
        splitter.split(inputs)
    ):
        if fold_idx == model_index:
            chosen = (
                inputs[train_index],
                outputs[train_index],
            )
            break

    if chosen is None:
        raise ValueError(
            f"MODEL_INDEX={model_index} is invalid for 5 folds."
        )

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

    # Match the training source behavior.
    return (
        x_train.astype("float32"),
        y_train.astype("float32"),
    )


# =============================================================================
# CHECKPOINT DISCOVERY
# =============================================================================

def discover_checkpoints(model_dir: Path, model_index: int):
    """
    Detects checkpoint names such as:
        k_model_epochs_1000_3.pth
        K_model_epochs_1000_3.pth

    Final models like k_model_3.pth are intentionally not included because
    H_R is intended as a checkpoint-vs-epoch diagnostic.
    """

    regex = re.compile(
        rf"^[kK]_model_epochs_(\d+)_{model_index}\.pth$"
    )

    iterator = (
        model_dir.rglob("*.pth")
        if RECURSIVE_MODEL_SEARCH
        else model_dir.glob("*.pth")
    )

    checkpoints = []

    for path in iterator:
        match = regex.match(path.name)
        if match:
            epoch = int(match.group(1))
            checkpoints.append((epoch, path))

    checkpoints.sort(key=lambda x: x[0])

    return checkpoints


# =============================================================================
# MODEL LOADING
# =============================================================================

def load_model(path: Path):
    """
    Handles full torch.save(model) checkpoints.
    """

    try:
        model = torch.load(
            path,
            map_location=DEVICE,
            weights_only=False,
        )
    except TypeError:
        model = torch.load(
            path,
            map_location=DEVICE,
        )

    if isinstance(model, nn.DataParallel):
        model = model.module

    if not isinstance(model, nn.Module):
        raise TypeError(
            f"{path} did not contain a full torch.nn.Module.\n"
            "If your checkpoints contain state_dict only, this loader must "
            "be adapted."
        )

    model = model.to(DEVICE)
    model.eval()

    return model


def infer_model_dtype(model):
    for p in model.parameters():
        return p.dtype
    return torch.float32


# =============================================================================
# LOSSES
# =============================================================================

def compute_losses(model, x, y, means_t, stds_t):
    """
    MSE:
        mean((y - y_pred)^2)

    Raw mass regularization:
        mean((sum(true species) - sum(pred species))^2)

    Temperature column is excluded from the mass term.
    """

    y_pred = model(x)

    mse = torch.mean(
        torch.square(y - y_pred)
    )

    pred_rescaled = (
        y_pred[:, 1:] * stds_t[1:]
        + means_t[1:]
    )

    true_rescaled = (
        y[:, 1:] * stds_t[1:]
        + means_t[1:]
    )

    pred_mass = torch.exp(pred_rescaled)
    true_mass = torch.exp(true_rescaled)

    mass_raw = torch.square(
        torch.sum(true_mass, dim=1)
        - torch.sum(pred_mass, dim=1)
    ).mean()

    mass_weighted = BETA * mass_raw

    # Preserve the same float behavior as the original training code.
    return (
        mse.float(),
        mass_raw.float(),
        mass_weighted.float(),
    )


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


def tensor_l2_norm(g):
    if g is None:
        return 0.0
    return float(
        torch.linalg.vector_norm(
            g.detach().reshape(-1),
            ord=2,
        ).cpu().item()
    )


def joint_l2_norm(grads):
    """
    Joint L2 norm for a collection of tensors:
        sqrt(sum_i ||g_i||_2^2)
    """
    total_sq = 0.0

    for g in grads:
        if g is None:
            continue

        total_sq += float(
            torch.sum(
                g.detach().double() ** 2
            ).cpu().item()
        )

    return float(np.sqrt(total_sq))


def get_linear_layer_parameter_groups(model, params):
    """
    Groups weight+bias belonging to each nn.Linear layer.

    Returns:
        [
            {
                "layer_index": 1,
                "layer_name": "...",
                "param_indices": [...]
            },
            ...
        ]
    """

    param_id_to_idx = {
        id(p): i
        for i, p in enumerate(params)
    }

    groups = []

    layer_counter = 0

    for module_name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue

        layer_counter += 1

        indices = []

        if module.weight is not None:
            idx = param_id_to_idx.get(
                id(module.weight)
            )
            if idx is not None:
                indices.append(idx)

        if module.bias is not None:
            idx = param_id_to_idx.get(
                id(module.bias)
            )
            if idx is not None:
                indices.append(idx)

        groups.append({
            "layer_index": layer_counter,
            "layer_name": (
                module_name
                if module_name
                else f"Linear_{layer_counter}"
            ),
            "param_indices": indices,
        })

    if not groups:
        raise RuntimeError(
            "No nn.Linear layers were found in the loaded model."
        )

    return groups


# =============================================================================
# CHECKPOINT ANALYSIS
# =============================================================================

def analyze_checkpoint(
    epoch,
    checkpoint_path,
    x_np,
    y_np,
    means_np,
    stds_np,
):
    model = load_model(checkpoint_path)
    dtype = infer_model_dtype(model)

    x = torch.as_tensor(
        x_np,
        dtype=dtype,
        device=DEVICE,
    )

    y = torch.as_tensor(
        y_np,
        dtype=dtype,
        device=DEVICE,
    )

    means_t = torch.as_tensor(
        means_np,
        dtype=dtype,
        device=DEVICE,
    )

    stds_t = torch.as_tensor(
        stds_np,
        dtype=dtype,
        device=DEVICE,
    )

    mse, mass_raw, mass_weighted = compute_losses(
        model,
        x,
        y,
        means_t,
        stds_t,
    )

    params = [
        p
        for p in model.parameters()
        if p.requires_grad
    ]

    # Independent gradients.
    mse_grads = separate_gradients(
        mse,
        params,
        retain_graph=True,
    )

    mass_raw_grads = separate_gradients(
        mass_raw,
        params,
        retain_graph=True,
    )

    # Weighted gradients are calculated explicitly as a sanity check.
    mass_beta_grads = separate_gradients(
        mass_weighted,
        params,
        retain_graph=False,
    )

    layer_groups = get_linear_layer_parameter_groups(
        model,
        params,
    )

    rows = []

    for group in layer_groups:
        idxs = group["param_indices"]

        mse_norm = joint_l2_norm(
            [mse_grads[i] for i in idxs]
        )

        mass_raw_norm = joint_l2_norm(
            [mass_raw_grads[i] for i in idxs]
        )

        mass_beta_norm = joint_l2_norm(
            [mass_beta_grads[i] for i in idxs]
        )

        R_raw = mass_raw_norm / (mse_norm + EPS)

        R_beta = mass_beta_norm / (mse_norm + EPS)

        log10_R_raw = np.log10(
            max(R_raw, np.finfo(float).tiny)
        )

        log10_R_beta = np.log10(
            max(R_beta, np.finfo(float).tiny)
        )

        rows.append({
            "checkpoint_epoch": epoch,
            "checkpoint_file": checkpoint_path.name,
            "mode": MODE,
            "beta": BETA,
            "model_index": MODEL_INDEX,
            "layer_index": group["layer_index"],
            "layer_name": group["layer_name"],
            "mse_loss": float(
                mse.detach().cpu().item()
            ),
            "mass_loss_raw": float(
                mass_raw.detach().cpu().item()
            ),
            "mass_loss_weighted": float(
                mass_weighted.detach().cpu().item()
            ),
            "mse_grad_norm": mse_norm,
            "mass_grad_norm_raw": mass_raw_norm,
            "mass_grad_norm_beta": mass_beta_norm,
            "R_l_raw": R_raw,
            "log10_R_l_raw": log10_R_raw,
            "R_l_beta": R_beta,
            "log10_R_l_beta": log10_R_beta,
        })

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return rows


# =============================================================================
# H_R
# =============================================================================

def compute_Hr(layer_df):
    """
    H_R(epoch) = max_l log10(R_l_raw) - min_l log10(R_l_raw)

    Because beta is global:
        H_R_beta == H_R_raw
    up to numerical precision.
    """

    summary_rows = []

    for epoch, part in layer_df.groupby(
        "checkpoint_epoch"
    ):
        logR_raw = part[
            "log10_R_l_raw"
        ].to_numpy(dtype=float)

        logR_beta = part[
            "log10_R_l_beta"
        ].to_numpy(dtype=float)

        raw_min_idx = part[
            "log10_R_l_raw"
        ].idxmin()

        raw_max_idx = part[
            "log10_R_l_raw"
        ].idxmax()

        H_raw = float(
            np.max(logR_raw)
            - np.min(logR_raw)
        )

        H_beta = float(
            np.max(logR_beta)
            - np.min(logR_beta)
        )

        summary_rows.append({
            "checkpoint_epoch": int(epoch),
            "mode": MODE,
            "beta": BETA,
            "H_R_raw": H_raw,
            "H_R_beta": H_beta,
            "spread_factor_raw": 10.0 ** H_raw,
            "spread_factor_beta": 10.0 ** H_beta,
            "log10_R_raw_min": float(
                np.min(logR_raw)
            ),
            "log10_R_raw_max": float(
                np.max(logR_raw)
            ),
            "min_layer": part.loc[
                raw_min_idx,
                "layer_name",
            ],
            "max_layer": part.loc[
                raw_max_idx,
                "layer_name",
            ],
            "n_layers": len(part),
        })

    return (
        pd.DataFrame(summary_rows)
        .sort_values("checkpoint_epoch")
        .reset_index(drop=True)
    )


# =============================================================================
# PLOTS
# =============================================================================

def plot_logR(layer_df, output_dir):
    fig, ax = plt.subplots(figsize=(9, 5))

    for layer_name, part in layer_df.groupby(
        "layer_name"
    ):
        part = part.sort_values(
            "checkpoint_epoch"
        )

        ax.plot(
            part["checkpoint_epoch"],
            part["log10_R_l_raw"],
            marker="o",
            label=layer_name,
        )

    ax.axhline(
        0.0,
        linewidth=1.0,
        linestyle="--",
    )

    ax.set_xlabel("Autosave epoch")
    ax.set_ylabel(
        r"$\log_{10}(R_l)$"
    )
    ax.set_title(
        f"Raw layer-wise gradient ratio — {MODE}"
    )
    ax.legend()
    ax.grid(alpha=0.25)

    fig.tight_layout()

    fig.savefig(
        output_dir
        / "log10_Rl_raw_vs_epoch.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_Hr(hr_df, output_dir):
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(
        hr_df["checkpoint_epoch"],
        hr_df["H_R_raw"],
        marker="o",
    )

    ax.set_xlabel("Autosave epoch")
    ax.set_ylabel(
        r"$H_R = \max_l \log_{10}R_l - "
        r"\min_l \log_{10}R_l$"
    )
    ax.set_title(
        f"Gradient heterogeneity index — {MODE}"
    )
    ax.grid(alpha=0.25)

    fig.tight_layout()

    fig.savefig(
        output_dir / "Hr_vs_epoch.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================

def main():
    mode = MODE.lower().strip()

    if mode not in {
        "clipping",
        "fixed",
    }:
        raise ValueError(
            'MODE must be either "clipping" or "fixed".'
        )

    print("=" * 78)
    print(
        "CHEMICAL KINETICS — GENERAL R_l / H_R ANALYSIS"
    )
    print("=" * 78)
    print(f"MODE        : {mode}")
    print(f"BETA        : {BETA:g}")
    print(f"MODEL_INDEX : {MODEL_INDEX}")
    print(f"DEVICE      : {DEVICE}")
    print(f"DATASET     : {DATASET_ROOT}")
    print(f"MODEL DIR   : {MODEL_DIR}")
    print()

    if mode == "clipping":
        print(
            "Clipping mode: H_R is calculated from RAW "
            "pre-clipping MSE and mass gradients."
        )
    else:
        print(
            "Fixed mode: H_R is calculated from RAW gradients. "
            "The beta-scaled ratio is also reported."
        )

    print()

    # ---------------------------------------------------------
    # Dataset
    # ---------------------------------------------------------

    inputs, outputs, means, stds = reconstruct_dataset_50(
        DATASET_ROOT
    )

    x_train, y_train = select_training_fold(
        inputs,
        outputs,
        MODEL_INDEX,
    )

    print(
        f"Training samples used: {len(x_train)}"
    )

    # ---------------------------------------------------------
    # Checkpoints
    # ---------------------------------------------------------

    checkpoints = discover_checkpoints(
        MODEL_DIR,
        MODEL_INDEX,
    )

    if not checkpoints:
        raise FileNotFoundError(
            "\nNo checkpoint files were found.\n"
            f"Searched below:\n{MODEL_DIR}\n\n"
            "Expected names like:\n"
            f"k_model_epochs_1000_{MODEL_INDEX}.pth"
        )

    print(
        f"Checkpoints found: {len(checkpoints)}"
    )

    for epoch, path in checkpoints:
        print(
            f"  epoch {epoch:>6}: {path}"
        )

    # ---------------------------------------------------------
    # Output folder
    # ---------------------------------------------------------

    beta_tag = (
        f"{BETA:.0e}"
        .replace("+", "")
    )

    output_dir = (
        MODEL_DIR
        / f"Hr_analysis_{mode}_beta_{beta_tag}"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------
    # Analyze
    # ---------------------------------------------------------

    all_rows = []

    for epoch, checkpoint_path in checkpoints:
        print()
        print(
            f"Analyzing epoch {epoch}: "
            f"{checkpoint_path.name}"
        )

        rows = analyze_checkpoint(
            epoch,
            checkpoint_path,
            x_train,
            y_train,
            means,
            stds,
        )

        all_rows.extend(rows)

    layer_df = pd.DataFrame(all_rows)

    if layer_df.empty:
        raise RuntimeError(
            "No layer diagnostics were produced."
        )

    # ---------------------------------------------------------
    # H_R
    # ---------------------------------------------------------

    hr_df = compute_Hr(
        layer_df
    )

    # ---------------------------------------------------------
    # Sanity check
    # ---------------------------------------------------------

    expected = (
        np.abs(BETA)
        * layer_df["R_l_raw"].to_numpy()
    )

    actual = (
        layer_df["R_l_beta"].to_numpy()
    )

    denom = np.maximum(
        np.abs(expected),
        EPS,
    )

    relative_error = np.max(
        np.abs(actual - expected)
        / denom
    )

    print()
    print(
        "Max relative error in "
        "R_l_beta ≈ |beta| R_l_raw:"
    )
    print(
        f"  {relative_error:.3e}"
    )

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    layer_csv = (
        output_dir
        / "Rl_by_checkpoint_and_layer.csv"
    )

    hr_csv = (
        output_dir
        / "Hr_by_checkpoint.csv"
    )

    layer_df.to_csv(
        layer_csv,
        index=False,
    )

    hr_df.to_csv(
        hr_csv,
        index=False,
    )

    plot_logR(
        layer_df,
        output_dir,
    )

    plot_Hr(
        hr_df,
        output_dir,
    )

    # ---------------------------------------------------------
    # Overall H_R summary
    # ---------------------------------------------------------

    overall = pd.DataFrame({
        "mode": [mode],
        "beta": [BETA],
        "model_index": [MODEL_INDEX],
        "mean_H_R": [
            hr_df["H_R_raw"].mean()
        ],
        "median_H_R": [
            hr_df["H_R_raw"].median()
        ],
        "min_H_R": [
            hr_df["H_R_raw"].min()
        ],
        "max_H_R": [
            hr_df["H_R_raw"].max()
        ],
        "mean_spread_factor": [
            hr_df[
                "spread_factor_raw"
            ].mean()
        ],
        "median_spread_factor": [
            hr_df[
                "spread_factor_raw"
            ].median()
        ],
    })

    overall_csv = (
        output_dir
        / "Hr_overall_summary.csv"
    )

    overall.to_csv(
        overall_csv,
        index=False,
    )

    # ---------------------------------------------------------
    # Console output
    # ---------------------------------------------------------

    print()
    print("=" * 78)
    print("H_R RESULTS")
    print("=" * 78)

    print(
        hr_df[
            [
                "checkpoint_epoch",
                "H_R_raw",
                "spread_factor_raw",
                "min_layer",
                "max_layer",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print(
        f"Mean H_R   : "
        f"{hr_df['H_R_raw'].mean():.4f}"
    )
    print(
        f"Median H_R : "
        f"{hr_df['H_R_raw'].median():.4f}"
    )
    print(
        f"Min H_R    : "
        f"{hr_df['H_R_raw'].min():.4f}"
    )
    print(
        f"Max H_R    : "
        f"{hr_df['H_R_raw'].max():.4f}"
    )

    print()
    print(
        "Interpretation: 10**H_R is the factor "
        "between the largest and smallest layer-wise R_l."
    )

    print()
    print("Saved:")
    print(f"  {layer_csv}")
    print(f"  {hr_csv}")
    print(f"  {overall_csv}")
    print(
        f"  {output_dir / 'log10_Rl_raw_vs_epoch.png'}"
    )
    print(
        f"  {output_dir / 'Hr_vs_epoch.png'}"
    )


if __name__ == "__main__":
    main()
