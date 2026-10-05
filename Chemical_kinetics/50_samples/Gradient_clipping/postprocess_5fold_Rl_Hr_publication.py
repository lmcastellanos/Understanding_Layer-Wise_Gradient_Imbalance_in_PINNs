#!/usr/bin/env python3
"""
Publication-style post-processing of the five-fold chemical-kinetics
R_l / H_R diagnostics.

This script reads:
    five_fold_direct_Rl_Hr/Rl_all_folds_all_checkpoints.csv

It then recomputes the fold-averaged quantities from the saved raw diagnostics:

1) For each epoch and layer:
       mean_f[log10(R_l)]
   with ±1 sample standard deviation across the five folds.

2) For each fold and epoch:
       H_R = max_l log10(R_l) - min_l log10(R_l)

   Then:
       mean_f[H_R]
   with ±1 sample standard deviation across the five folds.

Publication presentation:
- R_l legend labels are simply Layer 1, Layer 2, ...
- R_l y-axis label is log10(R_l)
- H_R y-axis label is only H_R
- no plot titles
- 13 x 8 in
- 600 dpi
- large publication fonts
- dashed light grid
- top/right spines removed

The script intentionally recalculates H_R from the saved layer-wise values,
rather than relying on a previously generated H_R summary.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# USER SETTINGS
# =============================================================================

ROOT_DIR = Path(__file__).resolve().parent

INPUT_CSV = (
    ROOT_DIR
    / "five_fold_direct_Rl_Hr"
    / "Rl_all_folds_all_checkpoints.csv"
)

OUTPUT_FOLDER = (
    ROOT_DIR
    / "five_fold_direct_Rl_Hr"
    / "publication_postprocessing"
)

EXPECTED_FOLDS = [0, 1, 2, 3, 4]


# =============================================================================
# PUBLICATION STYLE
# =============================================================================

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 24,
    "axes.labelsize": 30,
    "xtick.labelsize": 30,
    "ytick.labelsize": 30,
    "legend.fontsize": 22,
    "figure.dpi": 400,
    "savefig.dpi": 600,
    "lines.linewidth": 3.0,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# =============================================================================
# LOAD + VALIDATE
# =============================================================================

def load_data(path: Path):
    if not path.is_file():
        raise FileNotFoundError(
            "Could not find the fold-wise R_l CSV:\n"
            f"{path}\n\n"
            "Run the direct five-fold autosave analysis first."
        )

    df = pd.read_csv(path)

    required = {
        "model_index",
        "checkpoint_epoch",
        "layer_index",
        "log10_R_l_raw",
    }

    missing = required.difference(df.columns)

    if missing:
        raise ValueError(
            "The input CSV is missing required columns:\n"
            + "\n".join(sorted(missing))
        )

    df = df.copy()

    df["model_index"] = pd.to_numeric(
        df["model_index"],
        errors="raise",
    ).astype(int)

    df["checkpoint_epoch"] = pd.to_numeric(
        df["checkpoint_epoch"],
        errors="raise",
    ).astype(int)

    df["layer_index"] = pd.to_numeric(
        df["layer_index"],
        errors="raise",
    ).astype(int)

    df["log10_R_l_raw"] = pd.to_numeric(
        df["log10_R_l_raw"],
        errors="raise",
    )

    return df


def validate_folds(df):
    folds = sorted(
        df["model_index"]
        .unique()
        .tolist()
    )

    if folds != EXPECTED_FOLDS:
        raise RuntimeError(
            "Expected exactly folds [0, 1, 2, 3, 4], but found:\n"
            f"{folds}"
        )

    # Keep only epochs available for all five folds.
    epochs_by_fold = {
        fold: set(
            df.loc[
                df["model_index"] == fold,
                "checkpoint_epoch",
            ].unique()
        )
        for fold in EXPECTED_FOLDS
    }

    common_epochs = sorted(
        set.intersection(
            *epochs_by_fold.values()
        )
    )

    if not common_epochs:
        raise RuntimeError(
            "There are no checkpoint epochs shared by all five folds."
        )

    df = df[
        df["checkpoint_epoch"].isin(
            common_epochs
        )
    ].copy()

    return df, common_epochs


# =============================================================================
# CALCULATIONS
# =============================================================================

def compute_logRl_summary(df):
    """
    Mean ± sample SD of log10(R_l) across folds,
    separately for each layer and checkpoint epoch.
    """
    summary = (
        df.groupby(
            [
                "checkpoint_epoch",
                "layer_index",
            ],
            as_index=False,
        )
        .agg(
            mean_log10_R_l=(
                "log10_R_l_raw",
                "mean",
            ),
            sd_log10_R_l=(
                "log10_R_l_raw",
                "std",
            ),
            min_log10_R_l=(
                "log10_R_l_raw",
                "min",
            ),
            max_log10_R_l=(
                "log10_R_l_raw",
                "max",
            ),
            n_folds=(
                "model_index",
                "nunique",
            ),
        )
        .sort_values(
            [
                "checkpoint_epoch",
                "layer_index",
            ]
        )
        .reset_index(drop=True)
    )

    bad = summary[
        summary["n_folds"]
        != len(EXPECTED_FOLDS)
    ]

    if not bad.empty:
        raise RuntimeError(
            "Some epoch/layer combinations do not contain all five folds:\n"
            + bad.to_string(index=False)
        )

    return summary


def compute_foldwise_Hr(df):
    """
    Correct H_R calculation:
        H_R(fold, epoch)
          = max_l log10(R_l)
          - min_l log10(R_l)

    This is computed INSIDE each fold first.
    """
    rows = []

    for (
        fold,
        epoch,
    ), part in df.groupby(
        [
            "model_index",
            "checkpoint_epoch",
        ],
        sort=True,
    ):
        values = part[
            "log10_R_l_raw"
        ].to_numpy(dtype=float)

        rows.append({
            "model_index":
                int(fold),
            "checkpoint_epoch":
                int(epoch),
            "H_R":
                float(
                    np.max(values)
                    - np.min(values)
                ),
        })

    return (
        pd.DataFrame(rows)
        .sort_values(
            [
                "checkpoint_epoch",
                "model_index",
            ]
        )
        .reset_index(drop=True)
    )


def compute_Hr_summary(foldwise_hr):
    """
    Mean ± sample SD of H_R across the five folds.
    """
    summary = (
        foldwise_hr.groupby(
            "checkpoint_epoch",
            as_index=False,
        )
        .agg(
            mean_H_R=(
                "H_R",
                "mean",
            ),
            sd_H_R=(
                "H_R",
                "std",
            ),
            min_H_R=(
                "H_R",
                "min",
            ),
            max_H_R=(
                "H_R",
                "max",
            ),
            n_folds=(
                "model_index",
                "nunique",
            ),
        )
        .sort_values(
            "checkpoint_epoch"
        )
        .reset_index(drop=True)
    )

    bad = summary[
        summary["n_folds"]
        != len(EXPECTED_FOLDS)
    ]

    if not bad.empty:
        raise RuntimeError(
            "Some H_R checkpoint averages do not contain all five folds:\n"
            + bad.to_string(index=False)
        )

    return summary


# =============================================================================
# PLOTS
# =============================================================================

def plot_logRl(summary, output_path):
    fig, ax = plt.subplots(
        figsize=(13, 8),
        dpi=400,
    )

    for layer_index, part in summary.groupby(
        "layer_index",
        sort=True,
    ):
        part = part.sort_values(
            "checkpoint_epoch"
        )

        x = part[
            "checkpoint_epoch"
        ].to_numpy(dtype=float)

        y = part[
            "mean_log10_R_l"
        ].to_numpy(dtype=float)

        sd = (
            part["sd_log10_R_l"]
            .fillna(0.0)
            .to_numpy(dtype=float)
        )

        ax.plot(
            x,
            y,
            linewidth=3.0,
            label=f"Layer {int(layer_index)}",
        )

        ax.fill_between(
            x,
            y - sd,
            y + sd,
            alpha=0.14,
        )

    ax.axhline(
        0.0,
        linestyle="--",
        linewidth=1.2,
    )

    ax.set_xlabel(
        "Epoch",
        fontsize=30,
    )

    ax.set_ylabel(
        r"$\log_{10}(R_l)$",
        fontsize=30,
    )

    ax.tick_params(
        axis="both",
        labelsize=30,
    )

    ax.legend(
        fontsize=22,
        frameon=False,
    )

    ax.grid(
        True,
        linestyle="--",
        linewidth=0.8,
        alpha=0.45,
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=600,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_Hr(summary, output_path):
    fig, ax = plt.subplots(
        figsize=(13, 8),
        dpi=400,
    )

    x = summary[
        "checkpoint_epoch"
    ].to_numpy(dtype=float)

    y = summary[
        "mean_H_R"
    ].to_numpy(dtype=float)

    sd = (
        summary["sd_H_R"]
        .fillna(0.0)
        .to_numpy(dtype=float)
    )

    ax.plot(
        x,
        y,
        linewidth=3.0,
    )

    ax.fill_between(
        x,
        y - sd,
        y + sd,
        alpha=0.14,
    )

    ax.set_xlabel(
        "Epoch",
        fontsize=30,
    )

    # Keep the presentation deliberately simple.
    ax.set_ylabel(
        r"$H_R$",
        fontsize=30,
    )

    ax.tick_params(
        axis="both",
        labelsize=30,
    )

    ax.grid(
        True,
        linestyle="--",
        linewidth=0.8,
        alpha=0.45,
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=600,
        bbox_inches="tight",
    )

    plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 78)
    print("PUBLICATION POST-PROCESSING: FIVE-FOLD R_l / H_R")
    print("=" * 78)
    print(f"Input CSV : {INPUT_CSV}")
    print()

    df = load_data(
        INPUT_CSV
    )

    df, common_epochs = validate_folds(
        df
    )

    print(
        f"Common epochs across all five folds: {common_epochs}"
    )

    logRl_summary = compute_logRl_summary(
        df
    )

    foldwise_hr = compute_foldwise_Hr(
        df
    )

    hr_summary = compute_Hr_summary(
        foldwise_hr
    )

    OUTPUT_FOLDER.mkdir(
        parents=True,
        exist_ok=True,
    )

    logRl_csv = (
        OUTPUT_FOLDER
        / "log10_Rl_five_fold_mean_sd_publication.csv"
    )

    foldwise_hr_csv = (
        OUTPUT_FOLDER
        / "Hr_foldwise_recomputed.csv"
    )

    hr_csv = (
        OUTPUT_FOLDER
        / "Hr_five_fold_mean_sd_publication.csv"
    )

    logRl_summary.to_csv(
        logRl_csv,
        index=False,
    )

    foldwise_hr.to_csv(
        foldwise_hr_csv,
        index=False,
    )

    hr_summary.to_csv(
        hr_csv,
        index=False,
    )

    plot_logRl(
        logRl_summary,
        OUTPUT_FOLDER
        / "log10_Rl_five_fold_publication.png",
    )

    plot_Hr(
        hr_summary,
        OUTPUT_FOLDER
        / "Hr_five_fold_publication.png",
    )

    print()
    print("Saved:")
    print(f"  {logRl_csv}")
    print(f"  {foldwise_hr_csv}")
    print(f"  {hr_csv}")
    print(
        f"  {OUTPUT_FOLDER / 'log10_Rl_five_fold_publication.png'}"
    )
    print(
        f"  {OUTPUT_FOLDER / 'Hr_five_fold_publication.png'}"
    )
    print()
    print(
        "R_l plot legend: Layer 1, Layer 2, ... "
        "with mean ±1 SD across folds."
    )
    print(
        "H_R: recomputed inside each fold first, then averaged across folds."
    )


if __name__ == "__main__":
    main()
