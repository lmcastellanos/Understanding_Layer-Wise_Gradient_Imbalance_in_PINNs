#!/usr/bin/env python3

"""
Compare the Helmholtz results produced by:

1. Fixed regularization
2. Layer-wise gradient normalization

Expected folder structure:

helmholtz_test/
├── helmholtz_dataset.csv
├── compare_helmholtz_results.py
├── helmholtz_fixed_regularization_results/
│   ├── training_history.csv
│   ├── test_predictions.csv
│   └── metrics.csv
└── helmholtz_gradient_normalization_results/
    ├── training_history.csv
    ├── layer_gradient_scales.csv
    ├── test_predictions.csv
    └── metrics.csv

The script creates:

helmholtz_comparison_plots/
├── data_loss_comparison.png
├── pde_loss_comparison.png
├── solution_comparison.png
├── absolute_error_comparison.png
├── rmse_comparison.png
├── gradient_scale_history.png
├── gradient_norm_history.png
└── comparison_summary.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


FIXED_FOLDER = "helmholtz_fixed_regularization_results"
GRADNORM_FOLDER = "helmholtz_gradient_normalization_results"

DATASET_FILE = "helmholtz_dataset.csv"
OUTPUT_FOLDER = "helmholtz_comparison_plots"


def read_required_csv(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )
    return pd.read_csv(path)


def load_results(script_dir):

    dataset = read_required_csv(
        script_dir / DATASET_FILE
    )

    fixed_dir = (
        script_dir / FIXED_FOLDER
    )

    grad_dir = (
        script_dir / GRADNORM_FOLDER
    )

    fixed_history = read_required_csv(
        fixed_dir / "training_history.csv"
    )

    grad_history = read_required_csv(
        grad_dir / "training_history.csv"
    )

    fixed_predictions = read_required_csv(
        fixed_dir / "test_predictions.csv"
    )

    grad_predictions = read_required_csv(
        grad_dir / "test_predictions.csv"
    )

    fixed_metrics = read_required_csv(
        fixed_dir / "metrics.csv"
    )

    grad_metrics = read_required_csv(
        grad_dir / "metrics.csv"
    )

    scale_history = read_required_csv(
        grad_dir / "layer_gradient_scales.csv"
    )

    return {
        "dataset": dataset,
        "fixed_history": fixed_history,
        "grad_history": grad_history,
        "fixed_predictions": fixed_predictions,
        "grad_predictions": grad_predictions,
        "fixed_metrics": fixed_metrics,
        "grad_metrics": grad_metrics,
        "scale_history": scale_history,
    }


def plot_data_loss(results, output_dir):

    fixed = results["fixed_history"]
    grad = results["grad_history"]

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.semilogy(
        fixed["epoch"],
        fixed["data_loss"],
        label="Fixed regularization",
        linewidth=1.8,
    )

    ax.semilogy(
        grad["epoch"],
        grad["data_loss"],
        label="Gradient normalization",
        linewidth=1.8,
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Data MSE loss")
    ax.set_title(
        "Helmholtz data-loss convergence"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        output_dir
        / "data_loss_comparison.png",
        dpi=300,
    )

    plt.close(fig)


def plot_pde_loss(results, output_dir):

    fixed = results["fixed_history"]
    grad = results["grad_history"]

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.semilogy(
        fixed["epoch"],
        fixed["pde_loss"],
        label="Fixed regularization",
        linewidth=1.8,
    )

    ax.semilogy(
        grad["epoch"],
        grad["pde_loss"],
        label="Gradient normalization",
        linewidth=1.8,
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("PDE residual loss")
    ax.set_title(
        "Helmholtz PDE-loss convergence"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        output_dir
        / "pde_loss_comparison.png",
        dpi=300,
    )

    plt.close(fig)


def plot_solution(results, output_dir):

    dataset = (
        results["dataset"]
        .sort_values("x")
    )

    fixed = (
        results["fixed_predictions"]
        .sort_values("x")
    )

    grad = (
        results["grad_predictions"]
        .sort_values("x")
    )

    fig, ax = plt.subplots(
        figsize=(9, 5.5)
    )

    ax.plot(
        dataset["x"],
        dataset["u"],
        label="Exact solution",
        linewidth=2.2,
    )

    ax.scatter(
        fixed["x"],
        fixed["u_pred"],
        label="Fixed regularization",
        s=13,
        alpha=0.70,
    )

    ax.scatter(
        grad["x"],
        grad["u_pred"],
        label="Gradient normalization",
        s=13,
        alpha=0.70,
        marker="x",
    )

    ax.set_xlabel("x")
    ax.set_ylabel("u(x)")
    ax.set_title(
        "Helmholtz solution comparison"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        output_dir
        / "solution_comparison.png",
        dpi=300,
    )

    plt.close(fig)


def plot_absolute_error(results, output_dir):

    fixed = (
        results["fixed_predictions"]
        .sort_values("x")
    )

    grad = (
        results["grad_predictions"]
        .sort_values("x")
    )

    fig, ax = plt.subplots(
        figsize=(9, 5.5)
    )

    fixed_error = np.maximum(
        fixed["absolute_error"].to_numpy(),
        1e-16,
    )

    grad_error = np.maximum(
        grad["absolute_error"].to_numpy(),
        1e-16,
    )

    ax.semilogy(
        fixed["x"],
        fixed_error,
        ".",
        label="Fixed regularization",
        markersize=5,
    )

    ax.semilogy(
        grad["x"],
        grad_error,
        "x",
        label="Gradient normalization",
        markersize=4,
    )

    ax.set_xlabel("x")
    ax.set_ylabel("Absolute error")
    ax.set_title(
        "Helmholtz absolute test error"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        output_dir
        / "absolute_error_comparison.png",
        dpi=300,
    )

    plt.close(fig)


def plot_rmse(results, output_dir):

    fixed_rmse = float(
        results["fixed_metrics"][
            "test_RMSE"
        ].iloc[0]
    )

    grad_rmse = float(
        results["grad_metrics"][
            "test_RMSE"
        ].iloc[0]
    )

    labels = [
        "Fixed\nregularization",
        "Gradient\nnormalization",
    ]

    values = [
        fixed_rmse,
        grad_rmse,
    ]

    fig, ax = plt.subplots(
        figsize=(6.5, 5)
    )

    bars = ax.bar(
        labels,
        values,
    )

    ax.set_ylabel("Test RMSE")
    ax.set_title(
        "Helmholtz test-RMSE comparison"
    )

    ax.grid(
        True,
        axis="y",
        alpha=0.3,
    )

    ymax = max(values)

    if ymax > 0:
        ax.set_ylim(
            0,
            ymax * 1.25,
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
        output_dir
        / "rmse_comparison.png",
        dpi=300,
    )

    plt.close(fig)


def plot_gradient_scale_history(
    results,
    output_dir,
):

    scale_history = (
        results["scale_history"]
    )

    fig, ax = plt.subplots(
        figsize=(9, 5.5)
    )

    for layer in sorted(
        scale_history["layer"].unique()
    ):

        subset = scale_history[
            scale_history["layer"] == layer
        ]

        ax.plot(
            subset["epoch"],
            subset["scale"],
            label=f"Layer {layer}",
            linewidth=1.5,
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel(
        "Regularization gradient scale"
    )

    ax.set_ylim(
        0,
        1.05,
    )

    ax.set_title(
        "Layer-wise gradient-normalization scales"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        output_dir
        / "gradient_scale_history.png",
        dpi=300,
    )

    plt.close(fig)


def plot_gradient_norm_history(
    results,
    output_dir,
):

    scale_history = (
        results["scale_history"]
    )

    fig, ax = plt.subplots(
        figsize=(9, 5.5)
    )

    grouped = (
        scale_history
        .groupby("epoch")
        .agg({
            "data_grad_norm": "mean",
            "regularization_grad_norm": "mean",
            "scaled_regularization_grad_norm": "mean",
        })
        .reset_index()
    )

    ax.semilogy(
        grouped["epoch"],
        np.maximum(
            grouped["data_grad_norm"],
            1e-20,
        ),
        label="Data gradient norm",
        linewidth=1.7,
    )

    ax.semilogy(
        grouped["epoch"],
        np.maximum(
            grouped["regularization_grad_norm"],
            1e-20,
        ),
        label="Raw regularization gradient norm",
        linewidth=1.7,
    )

    ax.semilogy(
        grouped["epoch"],
        np.maximum(
            grouped[
                "scaled_regularization_grad_norm"
            ],
            1e-20,
        ),
        label="Scaled regularization gradient norm",
        linewidth=1.7,
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel(
        "Mean layer gradient norm"
    )

    ax.set_title(
        "Gradient magnitude comparison"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        output_dir
        / "gradient_norm_history.png",
        dpi=300,
    )

    plt.close(fig)


def save_summary(results, output_dir):

    fixed_metrics = (
        results["fixed_metrics"]
    )

    grad_metrics = (
        results["grad_metrics"]
    )

    fixed_rmse = float(
        fixed_metrics["test_RMSE"].iloc[0]
    )

    grad_rmse = float(
        grad_metrics["test_RMSE"].iloc[0]
    )

    fixed_mse = float(
        fixed_metrics["test_MSE"].iloc[0]
    )

    grad_mse = float(
        grad_metrics["test_MSE"].iloc[0]
    )

    summary = pd.DataFrame([
        {
            "method": "fixed_regularization",
            "test_MSE": fixed_mse,
            "test_RMSE": fixed_rmse,
        },
        {
            "method": "gradient_normalization",
            "test_MSE": grad_mse,
            "test_RMSE": grad_rmse,
        },
    ])

    summary.to_csv(
        output_dir
        / "comparison_summary.csv",
        index=False,
    )

    print("\n================================")
    print("Helmholtz comparison")
    print("================================")

    print(
        f"Fixed regularization RMSE: "
        f"{fixed_rmse:.6e}"
    )

    print(
        f"Gradient normalization RMSE: "
        f"{grad_rmse:.6e}"
    )

    if fixed_rmse > 0 and grad_rmse > 0:

        if grad_rmse < fixed_rmse:

            improvement = (
                fixed_rmse
                - grad_rmse
            ) / fixed_rmse * 100.0

            print(
                "Gradient normalization "
                f"has {improvement:.2f}% "
                "lower RMSE."
            )

        elif fixed_rmse < grad_rmse:

            improvement = (
                grad_rmse
                - fixed_rmse
            ) / grad_rmse * 100.0

            print(
                "Fixed regularization "
                f"has {improvement:.2f}% "
                "lower RMSE."
            )

        else:

            print(
                "Both methods have "
                "the same RMSE."
            )


def main():

    script_dir = (
        Path(__file__)
        .resolve()
        .parent
    )

    output_dir = (
        script_dir
        / OUTPUT_FOLDER
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = load_results(
        script_dir
    )

    plot_data_loss(
        results,
        output_dir,
    )

    plot_pde_loss(
        results,
        output_dir,
    )

    plot_solution(
        results,
        output_dir,
    )

    plot_absolute_error(
        results,
        output_dir,
    )

    plot_rmse(
        results,
        output_dir,
    )

    plot_gradient_scale_history(
        results,
        output_dir,
    )

    plot_gradient_norm_history(
        results,
        output_dir,
    )

    save_summary(
        results,
        output_dir,
    )

    print(
        f"\nPlots saved in:\n{output_dir}"
    )


if __name__ == "__main__":
    main()
