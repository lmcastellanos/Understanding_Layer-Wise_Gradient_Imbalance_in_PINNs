#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parent
EXPECTED_FOLDS = [0, 1, 2, 3, 4]
LOGR_COLUMN = "log10_R_l_raw"
OUTPUT_FOLDER = "five_fold_Rl_Hr_average"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 24,
    "axes.labelsize": 30,
    "xtick.labelsize": 30,
    "ytick.labelsize": 30,
    "legend.fontsize": 21,
    "figure.dpi": 400,
    "lines.linewidth": 3.0,
    "savefig.dpi": 600,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def discover_rl_csvs(root: Path):
    paths = []
    for path in sorted(root.rglob("Rl_by_checkpoint_and_layer.csv")):
        if OUTPUT_FOLDER in path.parts:
            continue
        try:
            df = pd.read_csv(path, nrows=5)
        except Exception:
            continue
        required = {"checkpoint_epoch", "model_index", LOGR_COLUMN}
        if required.issubset(df.columns):
            paths.append(path)
    return paths


def load_all(paths):
    frames = []
    for path in paths:
        df = pd.read_csv(path)
        required = {"checkpoint_epoch", "model_index", LOGR_COLUMN}
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
        if "layer_index" not in df.columns and "layer_name" not in df.columns:
            raise ValueError(f"No layer identifier found in {path}")
        df = df.copy()
        df["source_csv"] = str(path)
        frames.append(df)
        folds = sorted(pd.to_numeric(df["model_index"], errors="coerce").dropna().astype(int).unique())
        print(f"Loaded {path} | folds={folds}")
    if not frames:
        raise FileNotFoundError(f"No compatible Rl_by_checkpoint_and_layer.csv below {ROOT_DIR}")
    df = pd.concat(frames, ignore_index=True)
    df["model_index"] = pd.to_numeric(df["model_index"], errors="raise").astype(int)
    df["checkpoint_epoch"] = pd.to_numeric(df["checkpoint_epoch"], errors="raise").astype(int)
    df[LOGR_COLUMN] = pd.to_numeric(df[LOGR_COLUMN], errors="raise")
    return df


def choose_layer_cols(df):
    if "layer_index" in df.columns and df["layer_index"].notna().any():
        cols = ["layer_index"]
        if "layer_name" in df.columns:
            cols.append("layer_name")
        return cols
    return ["layer_name"]


def validate(df, layer_cols):
    found = sorted(df["model_index"].unique().tolist())
    missing = [f for f in EXPECTED_FOLDS if f not in found]
    if missing:
        raise RuntimeError(f"Missing folds: {missing}. Found: {found}")
    df = df[df["model_index"].isin(EXPECTED_FOLDS)].copy()
    key = ["model_index", "checkpoint_epoch", *layer_cols]
    dup = df.groupby(key, dropna=False).size().reset_index(name="n")
    dup = dup[dup["n"] > 1]
    if not dup.empty:
        raise RuntimeError(
            "Duplicate fold/epoch/layer rows found. Point ROOT_DIR to one configuration only.\n"
            + dup.head(20).to_string(index=False)
        )
    return df


def average_logR(df, layer_cols):
    group_cols = ["checkpoint_epoch", *layer_cols]
    return (
        df.groupby(group_cols, dropna=False)[LOGR_COLUMN]
        .agg(
            mean_log10_R_l="mean",
            sd_log10_R_l="std",
            min_log10_R_l="min",
            max_log10_R_l="max",
            n_folds="count",
        )
        .reset_index()
        .sort_values(group_cols)
        .reset_index(drop=True)
    )


def foldwise_Hr(df):
    rows = []
    for (fold, epoch), part in df.groupby(["model_index", "checkpoint_epoch"], sort=True):
        vals = part[LOGR_COLUMN].to_numpy(float)
        rows.append({
            "model_index": int(fold),
            "checkpoint_epoch": int(epoch),
            "H_R": float(vals.max() - vals.min()),
            "log10_R_min": float(vals.min()),
            "log10_R_max": float(vals.max()),
            "n_layers": int(len(vals)),
        })
    return pd.DataFrame(rows).sort_values(["checkpoint_epoch", "model_index"]).reset_index(drop=True)


def average_Hr(fold_hr, mean_logR):
    avg = (
        fold_hr.groupby("checkpoint_epoch")["H_R"]
        .agg(mean_H_R="mean", sd_H_R="std", min_H_R="min", max_H_R="max", n_folds="count")
        .reset_index()
        .sort_values("checkpoint_epoch")
        .reset_index(drop=True)
    )
    rows = []
    for epoch, part in mean_logR.groupby("checkpoint_epoch", sort=True):
        vals = part["mean_log10_R_l"].to_numpy(float)
        rows.append({
            "checkpoint_epoch": int(epoch),
            "H_R_of_mean_logR": float(vals.max() - vals.min()),
        })
    return avg.merge(pd.DataFrame(rows), on="checkpoint_epoch", how="left", validate="one_to_one")


def plot_logR(summary, layer_cols, out):
    fig, ax = plt.subplots(figsize=(13, 8), dpi=400)
    group_key = "layer_name" if "layer_name" in layer_cols else "layer_index"
    for layer, part in summary.groupby(group_key, sort=True):
        part = part.sort_values("checkpoint_epoch")
        x = part["checkpoint_epoch"].to_numpy(float)
        y = part["mean_log10_R_l"].to_numpy(float)
        sd = part["sd_log10_R_l"].fillna(0).to_numpy(float)
        label = str(layer) if group_key == "layer_name" else f"Layer {int(layer)}"
        ax.plot(x, y, label=label)
        ax.fill_between(x, y - sd, y + sd, alpha=0.14)
    ax.axhline(0, linestyle="--", linewidth=1.2)
    ax.set_xlabel("Epoch", fontsize=30)
    ax.set_ylabel(r"Mean $\log_{10}(R_l)$", fontsize=30)
    ax.tick_params(axis="both", labelsize=30)
    ax.legend(frameon=False, fontsize=21)
    ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.45)
    fig.tight_layout()
    fig.savefig(out, dpi=600, bbox_inches="tight")
    plt.close(fig)


def plot_Hr(summary, out):
    fig, ax = plt.subplots(figsize=(13, 8), dpi=400)
    x = summary["checkpoint_epoch"].to_numpy(float)
    y = summary["mean_H_R"].to_numpy(float)
    sd = summary["sd_H_R"].fillna(0).to_numpy(float)
    ax.plot(x, y, label=r"Mean $H_R$ across folds")
    ax.fill_between(x, y - sd, y + sd, alpha=0.14, label=r"$\pm1$ SD across folds")
    ax.plot(x, summary["H_R_of_mean_logR"].to_numpy(float), linestyle="--", linewidth=2.5,
            label=r"$H_R$ of mean $\log_{10}(R_l)$")
    ax.set_xlabel("Epoch", fontsize=30)
    ax.set_ylabel(r"$H_R$ [decades]", fontsize=30)
    ax.tick_params(axis="both", labelsize=30)
    ax.legend(frameon=False, fontsize=21)
    ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.45)
    fig.tight_layout()
    fig.savefig(out, dpi=600, bbox_inches="tight")
    plt.close(fig)


def main():
    root = ROOT_DIR.expanduser().resolve()
    paths = discover_rl_csvs(root)
    print(f"Found {len(paths)} compatible R_l CSV files below {root}")
    df = load_all(paths)
    layer_cols = choose_layer_cols(df)
    df = validate(df, layer_cols)

    logR = average_logR(df, layer_cols)
    fold_hr = foldwise_Hr(df)
    hr = average_Hr(fold_hr, logR)

    bad_logR = logR[logR["n_folds"] != len(EXPECTED_FOLDS)]
    bad_hr = hr[hr["n_folds"] != len(EXPECTED_FOLDS)]
    if not bad_logR.empty:
        raise RuntimeError("Some epoch/layer combinations do not contain all 5 folds:\n" + bad_logR.head(30).to_string(index=False))
    if not bad_hr.empty:
        raise RuntimeError("Some epochs do not contain all 5 fold H_R values:\n" + bad_hr.head(30).to_string(index=False))

    outdir = root / OUTPUT_FOLDER
    outdir.mkdir(parents=True, exist_ok=True)
    logR.to_csv(outdir / "five_fold_log10_Rl_by_epoch_and_layer.csv", index=False)
    fold_hr.to_csv(outdir / "five_fold_foldwise_Hr.csv", index=False)
    hr.to_csv(outdir / "five_fold_Hr_by_epoch.csv", index=False)
    plot_logR(logR, layer_cols, outdir / "five_fold_log10_Rl_mean_sd.png")
    plot_Hr(hr, outdir / "five_fold_Hr_mean_sd.png")

    print("\nPrimary definitions:")
    print("  log10(R_l): mean ± sample SD across folds for each epoch/layer")
    print("  H_R: compute H_R inside each fold first, then mean ± sample SD across folds")
    print("  H_R_of_mean_logR is also saved for comparison")
    print(f"\nSaved results in: {outdir}")


if __name__ == "__main__":
    main()
