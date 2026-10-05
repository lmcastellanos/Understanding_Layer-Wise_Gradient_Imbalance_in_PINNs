#!/usr/bin/env python3
"""
Five-fold all-species reconstruction: layer-wise gradient normalization vs
mass conservation (beta=1e-9), for element-wise and autoregressive prediction.

Change DATASET_FOLDER at the top (Datasets_20, Datasets_30, ...).
The corresponding model folder is inferred as 20_samples, 30_samples, etc.
"""

from pathlib import Path
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# =============================================================================
# USER SETTINGS
# =============================================================================
ROOT = Path.home() / "Alternative_trainings"

DATASET_FOLDER = "Datasets_30"       # <-- change this
EVALUATION_FOLDER = "interpolation" # interpolation/extrapolation/extrapolation_phi

# Set to None to infer e.g. Datasets_50 -> 50_samples.
MODEL_SAMPLE_FOLDER = None

LAYERWISE_SUBDIR = Path("Gradient_clipping") / "loss_layers-3" / "repeats_1"
MASS_SUBDIR = Path("Mass_conservation") / "1e-9" / "loss_layers-3" / "repeats_1"

N_FOLDS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOG_EPS = 1e-8

# =============================================================================
# PUBLICATION STYLE
# =============================================================================
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 24,
    "axes.labelsize": 30,
    "xtick.labelsize": 30,
    "ytick.labelsize": 30,
    "legend.fontsize": 20,
    "figure.dpi": 400,
    "savefig.dpi": 600,
    "lines.linewidth": 3.0,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# =============================================================================
# NETWORK / CHECKPOINT LOADING
# =============================================================================
class NN(nn.Module):
    def __init__(self, input_size, output_size, dropout_p, architecture_e,
                 min_val=None, max_val=None, alpha=0.001):
        super().__init__()
        layers = []
        for i, width in enumerate(architecture_e):
            nin = input_size if i == 0 else architecture_e[i-1]
            layers += [nn.Linear(nin, width), nn.Sigmoid(), nn.Dropout(dropout_p)]
        layers.append(nn.Linear(architecture_e[-1], output_size))
        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.encoder(x)


def clean_state_dict(sd):
    return {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}


def load_model(path, nvars):
    try:
        obj = torch.load(path, map_location=DEVICE, weights_only=False)
    except TypeError:
        obj = torch.load(path, map_location=DEVICE)

    if isinstance(obj, nn.DataParallel):
        obj = obj.module
    if isinstance(obj, nn.Module):
        obj = obj.to(DEVICE)
        obj.eval()
        return obj

    if not isinstance(obj, dict):
        raise TypeError(f"Unsupported checkpoint: {path}")

    sd = obj.get("model_state_dict", obj.get("state_dict", obj))
    sd = clean_state_dict(sd)

    # Infer hidden architecture from Linear weight matrices.
    weights = [v for v in sd.values() if torch.is_tensor(v) and v.ndim == 2]
    if len(weights) < 2:
        raise RuntimeError(f"Cannot infer architecture from {path}")
    hidden = [int(w.shape[0]) for w in weights[:-1]]

    model = NN(nvars, nvars, 0.0, hidden).to(DEVICE)
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model


def find_fold_models(folder):
    pattern = re.compile(r"^[kK]_model_([0-4])\.pth$")
    found = {}
    for path in folder.rglob("*.pth"):
        m = pattern.match(path.name)
        if m:
            fold = int(m.group(1))
            if fold in found:
                raise RuntimeError(f"Duplicate fold {fold} models:\n{found[fold]}\n{path}")
            found[fold] = path

    missing = sorted(set(range(N_FOLDS)) - set(found))
    if missing:
        raise FileNotFoundError(f"Missing folds {missing} under:\n{folder}")
    return [found[i] for i in range(N_FOLDS)]

# =============================================================================
# DATA
# =============================================================================
def sample_count():
    m = re.search(r"(\d+)$", DATASET_FOLDER)
    if not m:
        raise ValueError("DATASET_FOLDER must end in sample count, e.g. Datasets_50")
    return int(m.group(1))


def state_arrays(path):
    df = pd.read_csv(path)
    time = df.iloc[:, 1].to_numpy(float)
    T = df.iloc[:, 2].to_numpy(float).reshape(-1, 1)
    Y = df.iloc[:, 4:12].to_numpy(float)
    names = [str(df.columns[2])] + [str(x) for x in df.columns[4:12]]
    return time, np.concatenate([T, Y], axis=1), names


def find_training_file(root, T):
    pattern = re.compile(r"^State_space_cte_pressure_T(.+)_phi_(.+)\.csv$")
    for path in root.glob("State_space_cte_pressure_T*_phi_*.csv"):
        m = pattern.match(path.name)
        if not m:
            continue
        try:
            Tf, phif = float(m.group(1)), float(m.group(2))
        except ValueError:
            continue
        if np.isclose(Tf, float(T)) and np.isclose(phif, 1.0):
            return path
    raise FileNotFoundError(f"No training trajectory for T={T}, phi=1 under {root}")


def training_normalization(dataset_root, n):
    sampling = dataset_root / f"T_Sampling_uniform_{n}.csv"
    if not sampling.exists():
        matches = sorted(dataset_root.glob("T_Sampling_uniform_*.csv"))
        if len(matches) != 1:
            raise FileNotFoundError(f"Cannot identify sampling CSV in {dataset_root}")
        sampling = matches[0]

    table = pd.read_csv(sampling)
    temperatures = (table["T[K]"].to_numpy()
                    if "T[K]" in table.columns else table.iloc[:, 0].to_numpy())

    chunks = []
    for T in temperatures:
        _, physical, _ = state_arrays(find_training_file(dataset_root, T))
        chunks.append(np.log(physical + LOG_EPS))

    all_log = np.concatenate(chunks, axis=0)
    mean = all_log.mean(axis=0)
    std = all_log.std(axis=0, ddof=0)
    std[std == 0] = 1.0
    return mean, std, sampling


def prepare_case(path, mean, std):
    time, physical, names = state_arrays(path)
    z = (np.log(physical + LOG_EPS) - mean) / std
    current = z[:-1].astype("float32")
    reference = physical[1:]
    return current, reference, time[1:] * 1e3, names


def inverse(z, mean, std):
    return np.exp(z * std + mean)

# =============================================================================
# PREDICTIONS
# =============================================================================
@torch.no_grad()
def elementwise(model, x):
    return model(torch.from_numpy(x).to(DEVICE)).cpu().numpy()


@torch.no_grad()
def autoregressive(model, x0, nsteps):
    x = torch.tensor(x0, dtype=torch.float32, device=DEVICE).reshape(1, -1)
    out = []
    for _ in range(nsteps):
        x = model(x)
        out.append(x.cpu().numpy()[0])
    return np.asarray(out)


def all_fold_predictions(paths, current, mean, std, mode):
    pred = []
    for fold, path in enumerate(paths):
        print(f"      fold {fold}: {path}")
        model = load_model(path, current.shape[1])
        zhat = (elementwise(model, current) if mode == "element_wise"
                else autoregressive(model, current[0], len(current)))
        pred.append(inverse(zhat, mean, std))
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return np.stack(pred, axis=0)

# =============================================================================
# OUTPUT
# =============================================================================
def safe(s):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s))


def plot_all_species(t, ref, lw, mc, names, path):
    """
    Single 3x3 publication figure containing T + all 8 species.
    Each panel shows reference, five-fold mean, and +/-1 SD.
    A single shared legend is placed at the top.
    """
    lw_mean = lw.mean(axis=0)
    lw_sd = lw.std(axis=0, ddof=1)
    mc_mean = mc.mean(axis=0)
    mc_sd = mc.std(axis=0, ddof=1)

    fig, axes = plt.subplots(
        3, 3,
        figsize=(24, 18),
        dpi=400,
        sharex=False,
    )
    axes = axes.ravel()

    legend_handles = None
    legend_labels = None

    for j, (ax, name) in enumerate(zip(axes, names)):
        ax.plot(
            t, ref[:, j],
            color="black", lw=3.5, ls="-",
            label="Reference",
        )
        ax.plot(
            t, lw_mean[:, j],
            color="tab:blue", lw=3.0, ls="--",
            label="Layer-wise gradient normalization",
        )
        ax.fill_between(
            t,
            lw_mean[:, j] - lw_sd[:, j],
            lw_mean[:, j] + lw_sd[:, j],
            color="tab:blue",
            alpha=0.14,
        )
        ax.plot(
            t, mc_mean[:, j],
            color="tab:green", lw=3.0, ls="-.",
            label=r"Mass conservation $\beta=1\times10^{-9}$",
        )
        ax.fill_between(
            t,
            mc_mean[:, j] - mc_sd[:, j],
            mc_mean[:, j] + mc_sd[:, j],
            color="tab:green",
            alpha=0.14,
        )

        ax.set_ylabel(str(name), fontsize=25)
        ax.tick_params(axis="both", labelsize=22)
        ax.grid(True, ls="--", lw=0.8, alpha=0.45)

        # Bottom row only.
        if j >= 6:
            ax.set_xlabel("Time [ms]", fontsize=25)

        if legend_handles is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()

    # One shared legend for the complete figure.
    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=3,
        frameon=False,
        fontsize=22,
    )

    # No figure title: publication style.
    fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.94])

    fig.savefig(
        path,
        dpi=600,
        bbox_inches="tight",
    )
    plt.close(fig)


def save_predictions(path, t, ref, lw, mc, names):
    d = {"time_ms": t}
    for j, name in enumerate(names):
        tag = safe(name)
        d[f"{tag}_reference"] = ref[:, j]
        for f in range(N_FOLDS):
            d[f"{tag}_layerwise_fold_{f}"] = lw[f, :, j]
            d[f"{tag}_mass_fold_{f}"] = mc[f, :, j]
        d[f"{tag}_layerwise_mean"] = lw[:, :, j].mean(axis=0)
        d[f"{tag}_layerwise_sd"] = lw[:, :, j].std(axis=0, ddof=1)
        d[f"{tag}_mass_mean"] = mc[:, :, j].mean(axis=0)
        d[f"{tag}_mass_sd"] = mc[:, :, j].std(axis=0, ddof=1)
    pd.DataFrame(d).to_csv(path, index=False)


def main():
    n = sample_count()
    dataset_root = ROOT / DATASET_FOLDER
    model_sample = MODEL_SAMPLE_FOLDER or f"{n}_samples"
    model_root = ROOT / model_sample

    lw_dir = model_root / LAYERWISE_SUBDIR
    mc_dir = model_root / MASS_SUBDIR
    eval_dir = dataset_root / EVALUATION_FOLDER
    output = model_root / f"kfold_species_reconstruction_{DATASET_FOLDER}" / EVALUATION_FOLDER

    print(f"Dataset: {dataset_root}")
    print(f"Models : {model_root}")
    print(f"Cases  : {eval_dir}")

    mean, std, sampling = training_normalization(dataset_root, n)
    print(f"Normalization reconstructed from {sampling}")

    lw_models = find_fold_models(lw_dir)
    mc_models = find_fold_models(mc_dir)

    cases = sorted(eval_dir.glob("State_space_cte_pressure_T*_phi_*.csv"))
    if not cases:
        raise FileNotFoundError(f"No evaluation trajectories in {eval_dir}")

    output.mkdir(parents=True, exist_ok=True)

    for case in cases:
        print(f"\nCASE: {case.name}")
        current, ref, t, names = prepare_case(case, mean, std)
        case_out = output / case.stem

        for mode in ("element_wise", "autoregression"):
            print(f"  {mode}")
            lw = all_fold_predictions(lw_models, current, mean, std, mode)
            mc = all_fold_predictions(mc_models, current, mean, std, mode)

            mode_out = case_out / mode
            mode_out.mkdir(parents=True, exist_ok=True)
            save_predictions(mode_out / "five_fold_reconstruction.csv",
                             t, ref, lw, mc, names)

            plot_all_species(
                t,
                ref,
                lw,
                mc,
                names,
                mode_out / "all_species_reconstruction.png",
            )

    pd.DataFrame({
        "setting": ["dataset_folder", "model_sample_folder",
                    "evaluation_folder", "n_folds"],
        "value": [DATASET_FOLDER, model_sample, EVALUATION_FOLDER, N_FOLDS],
    }).to_csv(output / "analysis_settings.csv", index=False)

    print(f"\nSaved to:\n{output}")


if __name__ == "__main__":
    main()
