#!/usr/bin/env python3

import re
from pathlib import Path
from os import environ

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

DATASET_FOLDER = "Datasets_50"
GRADIENT_CLIPPING_FOLDER = "Gradient_clipping"
MASS_CONSERVATION_FOLDER = "Mass_conservation"
MASS_CONSTANT = "1e-9"
INTERPOLATION_FOLDER = "interpolation"
OUTPUT_FOLDER = "autoregression_response_comparison"

INTEREST_VECTOR = ["H2O", "O2", "H2", "T[K]", "OH", "HO2", "H2O2", "O", "H"]
STATE_PREFIX = "State_space_cte_pressure_T"
STATE_SUFFIX = "_phi_"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "figure.dpi": 300,
    "lines.linewidth": 2.0,
    "savefig.dpi": 600,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


class NN(nn.Module):
    def __init__(self, input_size, output_size, dropout_p, architecture_e, min_val=None, max_val=None, alpha=0.001):
        super(NN, self).__init__()
        encoder = []
        for i in range(len(architecture_e)):
            if i == 0:
                layer = nn.Linear(input_size, architecture_e[i])
            else:
                layer = nn.Linear(architecture_e[i - 1], architecture_e[i])
            encoder.extend([layer, nn.Sigmoid(), nn.Dropout(p=dropout_p)])
        encoder.append(nn.Linear(architecture_e[-1], output_size))
        self.encoder = nn.Sequential(*encoder)

    def forward(self, x):
        return self.encoder(x)


def detect_model(folder: Path):
    pattern = re.compile(r"^[Kk]_model_([0-4])\.pth$")
    found = []
    for path in sorted(folder.rglob("*.pth")):
        m = pattern.match(path.name)
        if m:
            found.append((int(m.group(1)), path))
    if len(found) == 0:
        raise FileNotFoundError(f"No k_model_n.pth/K_model_n.pth found under {folder}")
    if len(found) > 1:
        raise RuntimeError(f"More than one final model found under {folder}: {[str(p) for _, p in found]}")
    return found[0]


def rebuild_model(state_dict, input_size, output_size):
    model = NN(input_size, output_size, 0.0, [300, 300, 300])
    cleaned = {}
    for k, v in state_dict.items():
        cleaned[k[7:] if k.startswith("module.") else k] = v
    model.load_state_dict(cleaned, strict=True)
    model = model.to(DEVICE)
    model.eval()
    return model


def load_model(path, input_size, output_size):
    try:
        obj = torch.load(path, map_location=DEVICE)
    except Exception:
        obj = torch.load(path, map_location=DEVICE, weights_only=False)
    if isinstance(obj, nn.DataParallel):
        obj = obj.module
    if isinstance(obj, nn.Module):
        obj = obj.to(DEVICE)
        obj.eval()
        return obj
    if isinstance(obj, dict):
        return rebuild_model(obj, input_size, output_size)
    raise TypeError(f"Unsupported model format: {path}")


def transform_case(df):
    time = df.iloc[:, 1]
    temperature = df.iloc[:, 2]
    species = df.iloc[:, 4:12]
    data = pd.concat([temperature, species], axis=1)
    data = np.log(data + 1e-8)
    return data, data.columns.to_list(), time


def normalize_and_shift(data, means, stds):
    data = data.copy()
    for j in range(data.shape[1]):
        data.iloc[:, j] = (data.iloc[:, j] - means.iloc[0, j + 1]) / stds.iloc[0, j + 1]
    current = data.iloc[:-1, :].reset_index(drop=True)
    shifted = data.iloc[1:, :].reset_index(drop=True).add_suffix("_shift")
    return pd.concat([current, shifted], axis=1)


def autoregressive_prediction(model, first_input, n_steps):
    dtype = next(model.parameters()).dtype
    state = torch.tensor(first_input, dtype=dtype, device=DEVICE)
    out = np.zeros((n_steps, len(first_input)), dtype=float)
    with torch.no_grad():
        for i in range(n_steps):
            state = model(state)
            out[i] = state.detach().cpu().numpy()
    return out


def inverse_column(values, idx, means, stds):
    return np.exp(values * stds.iloc[0, idx + 1] + means.iloc[0, idx + 1])


def safe_name(text):
    return str(text).replace("/", "_").replace("\\", "_").replace("[", "").replace("]", "")


def plot_response(time_ms, ref, gc, mc, species, case_name, output_path):
    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.plot(time_ms, ref, color="black", linestyle="-", linewidth=2.4, label="Reference")
    ax.plot(time_ms, gc, color="tab:blue", linestyle="--", linewidth=2.0, label="Gradient clipping")
    ax.plot(time_ms, mc, color="tab:red", linestyle="-.", linewidth=2.0, label=r"Mass conservation $\lambda=10^{-9}$")
    ax.set_xlabel("Time [ms]")
    ax.set_ylabel(species)
    ax.set_title(f"{species} — {case_name}")
    ax.legend(frameon=False)
    #ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def main():
    script_dir = Path(__file__).resolve().parent
    dataset_root = Path(environ["HOME"]) / "Alternative_trainings" / DATASET_FOLDER
    interpolation_dir = dataset_root / INTERPOLATION_FOLDER
    means = pd.read_csv(dataset_root / "mean_values_log.csv")
    stds = pd.read_csv(dataset_root / "std_values_log.csv")

    _, gc_path = detect_model(script_dir / GRADIENT_CLIPPING_FOLDER)
    _, mc_path = detect_model(script_dir / MASS_CONSERVATION_FOLDER / MASS_CONSTANT)

    cases = sorted(interpolation_dir.glob(STATE_PREFIX + "*" + STATE_SUFFIX + "*.csv"))
    if not cases:
        raise FileNotFoundError(f"No interpolation state-space CSVs found in {interpolation_dir}")

    output_root = script_dir / OUTPUT_FOLDER
    output_root.mkdir(parents=True, exist_ok=True)

    for case_idx, csv_path in enumerate(cases):
        raw = pd.read_csv(csv_path)
        data_log, columns, time = transform_case(raw)
        dataset = normalize_and_shift(data_log, means, stds)
        input_size = len(columns)

        current = dataset.iloc[:, :input_size].to_numpy().astype("float32")
        targets = dataset.iloc[:, input_size:2 * input_size].to_numpy().astype("float32")
        n_steps = len(current)

        gc_model = load_model(gc_path, input_size, input_size)
        mc_model = load_model(mc_path, input_size, input_size)
        gc_pred = autoregressive_prediction(gc_model, current[0], n_steps)
        mc_pred = autoregressive_prediction(mc_model, current[0], n_steps)
        time_ms = time.iloc[1:].to_numpy() * 1e3

        case_dir = output_root / f"case_{case_idx}"
        case_dir.mkdir(parents=True, exist_ok=True)

        for species in INTEREST_VECTOR:
            if species not in columns:
                continue
            idx = columns.index(species)
            ref = inverse_column(targets[:, idx], idx, means, stds)
            gc = inverse_column(gc_pred[:, idx], idx, means, stds)
            mc = inverse_column(mc_pred[:, idx], idx, means, stds)
            plot_response(
                time_ms,
                ref,
                gc,
                mc,
                species,
                csv_path.stem,
                case_dir / f"{safe_name(species)}.png",
            )

    print(f"Saved comparison plots in: {output_root}")


if __name__ == "__main__":
    main()