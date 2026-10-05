#!/usr/bin/env python3

"""
Compare the results of the two Burgers-equation training strategies:

1. Fixed regularization:
       L = L_MSE + beta * L_PDE

2. Layer-wise gradient normalization:
       g_l = g_MSE,l + s_l * g_PDE,l

Expected folder structure:

burgers_test/
├── burgers_small_dataset.csv
├── train_burgers_fixed_regularization.py
├── train_burgers_gradient_normalization.py
├── burgers_fixed_regularization_results/
│   ├── training_history.csv
│   ├── test_predictions.csv
│   └── metrics.csv
└── burgers_gradient_normalization_results/
    ├── training_history.csv
    ├── layer_gradient_scales.csv
    ├── test_predictions.csv
    └── metrics.csv

The script creates:

burgers_comparison_plots/
├── training_mse_comparison.png
├── pde_loss_comparison.png
├── prediction_profiles.png
├── absolute_error_profiles.png
├── rmse_comparison.png
└── gradient_scale_history.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# SETTINGS
# ============================================================

FIXED_FOLDER = "burgers_fixed_regularization_results"
GRADNORM_FOLDER = "burgers_gradient_normalization_results"

DATASET_FILE = "burgers_small_dataset.csv"

OUTPUT_FOLDER = "burgers_comparison_plots"

# Times at which the full solution profiles are compared.
SELECTED_TIMES = [0.0, 0.25, 0.50, 0.75, 1.0]

# Tolerance used to match floating-point time values.
TIME_TOLERANCE = 1e-8


# ============================================================
# STYLE
# ============================================================

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "lines.linewidth": 2.0,
    "savefig.dpi": 400,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ============================================================
# HELPERS
# ============================================================

def require_file(path):
    if not path.is_file():
        raise FileNotFoundError(
            f"Required file was not found:\n{path}"
        )


def nearest_time(data, requested_time):
    available = np.sort(
        data["t"].unique()
    )

    index = np.argmin(
        np.abs(
            available - requested_time
        )
    )

    return float(
        available[index]
    )


def load_predictions(folder):
    path = folder / "test_predictions.csv"
    require_file(path)

    df = pd.read_csv(path)

    required = {
        "x",
        "t",
        "u",
        "u_pred",
    }

    missing = required - set(
        df.columns
    )

    if missing:
        raise ValueError(
            f"{path} is missing columns: {sorted(missing)}"
        )

    return df


# ============================================================
# LOSS PLOTS
# ============================================================

def plot_training_mse(
    fixed_history,
    grad_history,
    output_path,
):
    fig, ax = plt.subplots(
        figsize=(8, 5.5)
    )

    ax.semilogy(
        fixed_history["epoch"],
        fixed_history["mse_loss"],
        label="Fixed regularization",
    )

    ax.semilogy(
        grad_history["epoch"],
        grad_history["mse_loss"],
        linestyle="--",
        label="Gradient normalization",
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss")
    ax.set_title("Training data loss")

    ax.grid(
        alpha=0.2
    )
    ax.legend(
        frameon=False
    )

    fig.tight_layout()
    fig.savefig(
        output_path,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_pde_loss(
    fixed_history,
    grad_history,
    output_path,
):
    fig, ax = plt.subplots(
        figsize=(8, 5.5)
    )

    ax.semilogy(
        fixed_history["epoch"],
        fixed_history["pde_loss"],
        label="Fixed regularization",
    )

    ax.semilogy(
        grad_history["epoch"],
        grad_history["pde_loss"],
        linestyle="--",
        label="Gradient normalization",
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("PDE residual loss")
    ax.set_title("Burgers-equation residual during training")

    ax.grid(
        alpha=0.2
    )
    ax.legend(
        frameon=False
    )

    fig.tight_layout()
    fig.savefig(
        output_path,
        bbox_inches="tight",
    )
    plt.close(fig)


# ============================================================
# PROFILE COMPARISON
# ============================================================

def build_prediction_grid(
    full_dataset,
    prediction_df,
    selected_time,
):
    """
    The original training scripts used a random train/test split, so the
    test_predictions files contain only test locations.

    This function returns the reference full profile and the test prediction
    points nearest to the requested time.
    """

    actual_time = nearest_time(
        full_dataset,
        selected_time,
    )

    reference = full_dataset[
        np.isclose(
            full_dataset["t"],
            actual_time,
            atol=TIME_TOLERANCE,
        )
    ].sort_values("x")

    predicted = prediction_df[
        np.isclose(
            prediction_df["t"],
            actual_time,
            atol=TIME_TOLERANCE,
        )
    ].sort_values("x")

    return (
        actual_time,
        reference,
        predicted,
    )


def plot_prediction_profiles(
    full_dataset,
    fixed_predictions,
    grad_predictions,
    output_path,
):
    n_times = len(
        SELECTED_TIMES
    )

    fig, axes = plt.subplots(
        n_times,
        1,
        figsize=(8.5, 3.3 * n_times),
        sharex=True,
    )

    if n_times == 1:
        axes = [axes]

    for ax, requested_time in zip(
        axes,
        SELECTED_TIMES,
    ):
        (
            actual_time,
            reference,
            fixed_at_time,
        ) = build_prediction_grid(
            full_dataset,
            fixed_predictions,
            requested_time,
        )

        (
            _,
            _,
            grad_at_time,
        ) = build_prediction_grid(
            full_dataset,
            grad_predictions,
            requested_time,
        )

        ax.plot(
            reference["x"],
            reference["u"],
            label="Reference",
        )

        if len(
            fixed_at_time
        ) > 0:
            ax.scatter(
                fixed_at_time["x"],
                fixed_at_time["u_pred"],
                s=24,
                marker="o",
                label="Fixed regularization",
            )

        if len(
            grad_at_time
        ) > 0:
            ax.scatter(
                grad_at_time["x"],
                grad_at_time["u_pred"],
                s=28,
                marker="x",
                label="Gradient normalization",
            )

        ax.set_ylabel("u(x,t)")
        ax.set_title(
            f"t = {actual_time:.2f}"
        )

        ax.grid(
            alpha=0.2
        )

    axes[-1].set_xlabel("x")

    axes[0].legend(
        frameon=False,
        ncol=3,
    )

    fig.suptitle(
        "Burgers solution: reference vs network predictions",
        y=0.995,
    )

    fig.tight_layout()
    fig.savefig(
        output_path,
        bbox_inches="tight",
    )
    plt.close(fig)


# ============================================================
# ABSOLUTE ERROR PROFILE
# ============================================================

def plot_absolute_error_profiles(
    fixed_predictions,
    grad_predictions,
    output_path,
):
    n_times = len(
        SELECTED_TIMES
    )

    fig, axes = plt.subplots(
        n_times,
        1,
        figsize=(8.5, 3.2 * n_times),
        sharex=True,
    )

    if n_times == 1:
        axes = [axes]

    all_predictions = pd.concat(
        [
            fixed_predictions[["t"]],
            grad_predictions[["t"]],
        ]
    )

    for ax, requested_time in zip(
        axes,
        SELECTED_TIMES,
    ):
        actual_time = nearest_time(
            all_predictions,
            requested_time,
        )

        fixed_at_time = fixed_predictions[
            np.isclose(
                fixed_predictions["t"],
                actual_time,
                atol=TIME_TOLERANCE,
            )
        ].copy()

        grad_at_time = grad_predictions[
            np.isclose(
                grad_predictions["t"],
                actual_time,
                atol=TIME_TOLERANCE,
            )
        ].copy()

        fixed_at_time = fixed_at_time.sort_values(
            "x"
        )

        grad_at_time = grad_at_time.sort_values(
            "x"
        )

        if len(
            fixed_at_time
        ) > 0:
            fixed_error = np.abs(
                fixed_at_time["u_pred"]
                - fixed_at_time["u"]
            )

            ax.plot(
                fixed_at_time["x"],
                fixed_error,
                marker="o",
                markersize=3,
                label="Fixed regularization",
            )

        if len(
            grad_at_time
        ) > 0:
            grad_error = np.abs(
                grad_at_time["u_pred"]
                - grad_at_time["u"]
            )

            ax.plot(
                grad_at_time["x"],
                grad_error,
                marker="x",
                markersize=4,
                linestyle="--",
                label="Gradient normalization",
            )

        ax.set_yscale("log")
        ax.set_ylabel("|error|")
        ax.set_title(
            f"t = {actual_time:.2f}"
        )
        ax.grid(
            alpha=0.2
        )

    axes[-1].set_xlabel("x")

    axes[0].legend(
        frameon=False
    )

    fig.suptitle(
        "Absolute prediction error",
        y=0.995,
    )

    fig.tight_layout()
    fig.savefig(
        output_path,
        bbox_inches="tight",
    )
    plt.close(fig)


# ============================================================
# RMSE BAR GRAPH
# ============================================================

def plot_rmse(
    fixed_metrics,
    grad_metrics,
    output_path,
):
    fixed_rmse = float(
        fixed_metrics["test_RMSE"].iloc[0]
    )

    grad_rmse = float(
        grad_metrics["test_RMSE"].iloc[0]
    )

    methods = [
        "Fixed regularization",
        "Gradient normalization",
    ]

    values = [
        fixed_rmse,
        grad_rmse,
    ]

    fig, ax = plt.subplots(
        figsize=(7, 5)
    )

    bars = ax.bar(
        methods,
        values,
    )

    ax.set_ylabel("Test RMSE")
    ax.set_title(
        "Overall Burgers-equation test error"
    )

    ax.grid(
        axis="y",
        alpha=0.2,
    )

    for bar, value in zip(
        bars,
        values,
    ):
        ax.text(
            bar.get_x()
            + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.3e}",
            ha="center",
            va="bottom",
        )

    fig.tight_layout()
    fig.savefig(
        output_path,
        bbox_inches="tight",
    )
    plt.close(fig)


# ============================================================
# GRADIENT NORMALIZATION SCALE HISTORY
# ============================================================

def plot_gradient_scales(
    scale_history,
    output_path,
):
    fig, ax = plt.subplots(
        figsize=(8.5, 5.5)
    )

    for layer in sorted(
        scale_history["layer"].unique()
    ):
        subset = scale_history[
            scale_history["layer"]
            == layer
        ]

        ax.plot(
            subset["epoch"],
            subset["scale"],
            label=f"Layer {layer}",
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("PDE-gradient scaling factor")
    ax.set_title(
        "Layer-wise gradient-normalization factors"
    )

    ax.set_ylim(
        0,
        1.05,
    )

    ax.grid(
        alpha=0.2
    )

    ax.legend(
        frameon=False,
        ncol=2,
    )

    fig.tight_layout()
    fig.savefig(
        output_path,
        bbox_inches="tight",
    )
    plt.close(fig)


# ============================================================
# MAIN
# ============================================================

def main():

    script_dir = (
        Path(__file__)
        .resolve()
        .parent
    )

    fixed_dir = (
        script_dir
        / FIXED_FOLDER
    )

    grad_dir = (
        script_dir
        / GRADNORM_FOLDER
    )

    output_dir = (
        script_dir
        / OUTPUT_FOLDER
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Required files
    # --------------------------------------------------------

    dataset_path = (
        script_dir
        / DATASET_FILE
    )

    fixed_history_path = (
        fixed_dir
        / "training_history.csv"
    )

    grad_history_path = (
        grad_dir
        / "training_history.csv"
    )

    fixed_metrics_path = (
        fixed_dir
        / "metrics.csv"
    )

    grad_metrics_path = (
        grad_dir
        / "metrics.csv"
    )

    scale_path = (
        grad_dir
        / "layer_gradient_scales.csv"
    )

    for path in [
        dataset_path,
        fixed_history_path,
        grad_history_path,
        fixed_metrics_path,
        grad_metrics_path,
        scale_path,
    ]:
        require_file(
            path
        )

    # --------------------------------------------------------
    # Load results
    # --------------------------------------------------------

    full_dataset = pd.read_csv(
        dataset_path
    )

    fixed_history = pd.read_csv(
        fixed_history_path
    )

    grad_history = pd.read_csv(
        grad_history_path
    )

    fixed_predictions = load_predictions(
        fixed_dir
    )

    grad_predictions = load_predictions(
        grad_dir
    )

    fixed_metrics = pd.read_csv(
        fixed_metrics_path
    )

    grad_metrics = pd.read_csv(
        grad_metrics_path
    )

    scale_history = pd.read_csv(
        scale_path
    )

    # --------------------------------------------------------
    # Generate plots
    # --------------------------------------------------------

    plot_training_mse(
        fixed_history,
        grad_history,
        output_dir
        / "training_mse_comparison.png",
    )

    plot_pde_loss(
        fixed_history,
        grad_history,
        output_dir
        / "pde_loss_comparison.png",
    )

    plot_prediction_profiles(
        full_dataset,
        fixed_predictions,
        grad_predictions,
        output_dir
        / "prediction_profiles.png",
    )

    plot_absolute_error_profiles(
        fixed_predictions,
        grad_predictions,
        output_dir
        / "absolute_error_profiles.png",
    )

    plot_rmse(
        fixed_metrics,
        grad_metrics,
        output_dir
        / "rmse_comparison.png",
    )

    plot_gradient_scales(
        scale_history,
        output_dir
        / "gradient_scale_history.png",
    )

    # --------------------------------------------------------
    # Print numerical comparison
    # --------------------------------------------------------

    fixed_rmse = float(
        fixed_metrics[
            "test_RMSE"
        ].iloc[0]
    )

    grad_rmse = float(
        grad_metrics[
            "test_RMSE"
        ].iloc[0]
    )

    print(
        "\n=================================="
    )
    print(
        "Burgers training comparison"
    )
    print(
        "=================================="
    )

    print(
        f"Fixed regularization RMSE: "
        f"{fixed_rmse:.6e}"
    )

    print(
        f"Gradient normalization RMSE: "
        f"{grad_rmse:.6e}"
    )

    if grad_rmse < fixed_rmse:
        improvement = (
            100.0
            * (
                fixed_rmse
                - grad_rmse
            )
            / fixed_rmse
        )

        print(
            "Gradient normalization has "
            f"{improvement:.2f}% lower RMSE."
        )

    elif fixed_rmse < grad_rmse:
        improvement = (
            100.0
            * (
                grad_rmse
                - fixed_rmse
            )
            / grad_rmse
        )

        print(
            "Fixed regularization has "
            f"{improvement:.2f}% lower RMSE."
        )

    else:
        print(
            "The two methods have identical RMSE."
        )

    print(
        f"\nPlots saved in:\n{output_dir}"
    )


if __name__ == "__main__":
    main()
