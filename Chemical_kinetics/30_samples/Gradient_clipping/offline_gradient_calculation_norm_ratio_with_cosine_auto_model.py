import re
from pathlib import Path
from os.path import join
from os import environ

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

home = environ["HOME"]
folder1 = "Alternative_trainings"
folder2 = "Datasets_30"

phi_name = 1.0
n_autoregression = 1

grad_eps = 1e-12
use_global_norm_ratio = False

name_11 = "State_space_cte_pressure_T"
name_12 = "_phi_"
end = ".csv"

PLOTTING_COLUMNS = ["run_id", "epoch", "loss_part", "layer", "gradient"]
COSINE_COLUMNS = [
    "run_id",
    "epoch",
    "layer",
    "mse_norm",
    "reg_norm",
    "scaled_reg_norm",
    "combined_norm",
    "reg_scale",
    "reg_scaled",
    "cosine_mse_reg",
    "cosine_mse_scaled_reg",
    "cosine_mse_combined",
]


def hydrogen_log_transform_sandiego_cantera(cantera_species):
    cantera_temperature = cantera_species.iloc[:, 2]
    cantera_species_fractions = cantera_species.iloc[:, 4:12]

    n_columns_mass_fraction = np.shape(cantera_species_fractions)[1]
    data_all = pd.concat([cantera_temperature, cantera_species_fractions], axis=1)
    data_all = np.log(data_all + 1e-8)
    columns = data_all.columns.to_list()
    return data_all, n_columns_mass_fraction, columns


def z_score_data_transformation_time_shift(dataset, n_autoregression, columns):
    iterations = np.shape(dataset)[1]
    means = np.zeros(iterations)
    stds = np.zeros(iterations)

    dataset = dataset.copy()

    for j in range(iterations):
        means[j] = np.mean(dataset.iloc[:, j])
        stds[j] = np.std(dataset.iloc[:, j])
        if stds[j] == 0:
            stds[j] = 1.0
        dataset.iloc[:, j] = (dataset.iloc[:, j] - means[j]) / stds[j]

    if n_autoregression != 1:
        dataset_copy = dataset.copy()
        core_data = dataset_copy.iloc[1:-n_autoregression, 1:].copy()
        core_data.reset_index(drop=True, inplace=True)
        shifted_dfs = [core_data]

        for i in range(1, n_autoregression + 2):
            dataset_copy2 = dataset.copy()
            shifted_df = dataset_copy2.iloc[i:-n_autoregression + 1, :].copy()
            shifted_df.reset_index(drop=True, inplace=True)
            shifted_dfs.append(shifted_df)

        result_df = pd.concat(shifted_dfs, axis=1)
        dataset_out = result_df.iloc[:-n_autoregression, :]
    else:
        dataset_t0 = dataset.iloc[:-1, :].copy()
        dataset_shift = dataset.iloc[1:, :].copy()
        dataset_shift = dataset_shift.add_suffix("_shift")
        dataset_t0 = dataset_t0.reset_index(drop=True)
        dataset_shift = dataset_shift.reset_index(drop=True)
        dataset_out = pd.concat([dataset_t0, dataset_shift], axis=1)

    return dataset_out, means, stds


def build_dataset():
    condition_csv = join(home, folder1, folder2, "T_Sampling_uniform_30.csv")
    condition_inputs = pd.read_csv(condition_csv)

    dataset_copy = None
    n_columns_mass_fraction = None
    columns = None

    iterations = np.shape(condition_inputs)[0]
    for i in range(iterations):
        cantera_path = join(
            home,
            folder1,
            folder2,
            name_11 + str(round(condition_inputs.loc[i, "T[K]"], 3)) + name_12 + str(phi_name) + end,
        )
        cantera_species = pd.read_csv(cantera_path)
        cantera_species = pd.DataFrame(cantera_species)

        dataset, n_columns_mass_fraction, columns = hydrogen_log_transform_sandiego_cantera(cantera_species)

        if dataset_copy is None:
            dataset_copy = dataset
        else:
            dataset_copy = pd.concat([dataset_copy, dataset], axis=0)

    dataset_out, means, stds = z_score_data_transformation_time_shift(dataset_copy, n_autoregression, columns)

    input_size = n_columns_mass_fraction + 1
    output_start = 1 + n_columns_mass_fraction
    output_end = output_start + n_columns_mass_fraction + 1

    inputs = dataset_out.iloc[:, :input_size].to_numpy().astype("float32")
    outputs = dataset_out.iloc[:, output_start:output_end].to_numpy().astype("float32")

    return {
        "inputs": torch.from_numpy(inputs).to(device),
        "outputs": torch.from_numpy(outputs).to(device),
        "input_size": int(input_size),
        "output_size": int(input_size),
        "n_rows": int(inputs.shape[0]),
        "means": torch.from_numpy(means.astype("float32")),
        "stds": torch.from_numpy(stds.astype("float32")),
    }


class NN(nn.Module):
    def __init__(self, input_size, output_size, dropout_p, architecture_e, min_val=None, max_val=None):
        super(NN, self).__init__()

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


def loss_MSE_norm_ratio(y_pred, y_train, means, stds):
    mse = torch.mean(torch.square(y_train - y_pred))

    means = means.to(device=y_pred.device, dtype=y_pred.dtype)
    stds = stds.to(device=y_pred.device, dtype=y_pred.dtype)

    y_pred_rescaled = y_pred[:, 1:] * stds[1:] + means[1:]
    y_true_rescaled = y_train[:, 1:] * stds[1:] + means[1:]

    y_pred_rescaled = torch.exp(y_pred_rescaled)
    y_true_rescaled = torch.exp(y_true_rescaled)

    reg = torch.square(torch.sum(y_true_rescaled, dim=1) - torch.sum(y_pred_rescaled, dim=1)).mean()

    return mse.float(), reg.float()


def parse_model_file_info(path: Path):
    name = path.name

    m_epoch = re.match(r"k_model_epochs_(\d+)_(\d+)\.pth$", name)
    if m_epoch:
        return {"epoch": int(m_epoch.group(1)), "run_id": int(m_epoch.group(2)), "kind": "checkpoint"}

    m_final = re.match(r"[Kk]_model_(\d+)\.pth$", name)
    if m_final:
        run_id = int(m_final.group(1))
        if 0 <= run_id <= 4:
            return {"epoch": None, "run_id": run_id, "kind": "final"}
        return None

    return None


def infer_final_epoch_from_loss_csv(script_dir: Path, run_id: int):
    candidates = [
        script_dir / f"k_model_loss{run_id}.csv",
        script_dir / f"k_model_reg_loss{run_id}.csv",
        script_dir / f"k_model_grad_stats{run_id}.csv",
    ]

    lengths = []
    for csv_path in candidates:
        if not csv_path.exists():
            continue
        try:
            if csv_path.name.endswith("grad_stats" + str(run_id) + ".csv"):
                arr = pd.read_csv(csv_path)
                lengths.append(int(len(arr)))
            else:
                arr = np.loadtxt(csv_path, delimiter=",")
                arr = np.atleast_1d(arr)
                lengths.append(int(len(arr)))
        except Exception:
            continue

    if lengths:
        return max(lengths)
    return None


def find_model_files(script_dir: Path):
    """Detect exactly one final model k_model_n.pth/K_model_n.pth with n in [0, 4]."""
    candidates = []
    for path in sorted(script_dir.iterdir()):
        if not path.is_file():
            continue
        match = re.fullmatch(r"[Kk]_model_([0-4])\.pth", path.name)
        if match is None:
            continue
        run_id = int(match.group(1))
        epoch = infer_final_epoch_from_loss_csv(script_dir, run_id)
        if epoch is None:
            raise FileNotFoundError(
                f"Detected model '{path.name}', but its final epoch could not be inferred. "
                f"Expected k_model_loss{run_id}.csv, k_model_reg_loss{run_id}.csv, "
                f"or k_model_grad_stats{run_id}.csv."
            )
        candidates.append({
            "epoch": epoch, "run_id": run_id, "kind": "final", "path": path
        })
    if len(candidates) == 0:
        return []
    if len(candidates) > 1:
        names = ", ".join(item["path"].name for item in candidates)
        raise RuntimeError(
            "More than one final model was detected. Expected exactly one "
            f"k_model_n.pth/K_model_n.pth with n in [0, 4]. Found: {names}"
        )
    detected = candidates[0]
    print(
        f"Detected model: {detected['path'].name} "
        f"(model number/run_id={detected['run_id']}, epoch={detected['epoch']})"
    )
    return candidates


def get_linear_layer_name(param_name: str):
    clean_name = param_name.replace("module.", "")
    m = re.match(r"encoder\.(\d+)\.(weight|bias)$", clean_name)
    if not m:
        return "other"

    module_index = int(m.group(1))
    linear_positions = [0, 3, 6, 9]

    if module_index in linear_positions:
        linear_number = linear_positions.index(module_index) + 1
        if linear_number == len(linear_positions):
            return "output"
        return f"hidden_{linear_number}"

    return "other"


def rebuild_model_from_state_dict(state_dict, input_size, output_size):
    architecture_e = [300, 300, 300]
    dropout_p = 0.0

    model = NN(
        input_size=input_size,
        output_size=output_size,
        dropout_p=dropout_p,
        architecture_e=architecture_e,
        min_val=None,
        max_val=None,
    )

    cleaned_state = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            cleaned_state[k[len("module."):]] = v
        else:
            cleaned_state[k] = v

    model.load_state_dict(cleaned_state, strict=True)
    model = model.to(device)
    model.eval()
    return model


def load_model(model_path: Path, input_size: int, output_size: int):
    try:
        obj = torch.load(model_path, map_location=device)
        if isinstance(obj, nn.Module):
            model = obj
            if isinstance(model, nn.DataParallel):
                model = model.module
            model = model.to(device)
            model.eval()
            return model
        if isinstance(obj, dict):
            return rebuild_model_from_state_dict(obj, input_size, output_size)
    except Exception:
        pass

    obj = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(obj, nn.Module):
        model = obj
        if isinstance(model, nn.DataParallel):
            model = model.module
        model = model.to(device)
        model.eval()
        return model
    if isinstance(obj, dict):
        return rebuild_model_from_state_dict(obj, input_size, output_size)

    raise TypeError(f"Unsupported model content in file: {model_path}")


def global_grad_norm(grads, norm_type=2, device_override=None):
    valid = [g.reshape(-1) for g in grads if g is not None]
    if not valid:
        dev = device_override if device_override is not None else device
        return torch.tensor(0.0, device=dev)
    flat = torch.cat(valid)
    return torch.norm(flat, p=norm_type)


def compute_separate_grads(loss_mse, loss_reg, model):
    params = [p for p in model.parameters() if p.requires_grad]

    mse_grads = torch.autograd.grad(
        loss_mse,
        params,
        retain_graph=True,
        create_graph=False,
        allow_unused=True,
    )

    reg_grads = torch.autograd.grad(
        loss_reg,
        params,
        retain_graph=False,
        create_graph=False,
        allow_unused=True,
    )

    return params, mse_grads, reg_grads


def safe_zeros_like(reference, fallback_param):
    if reference is not None:
        return torch.zeros_like(reference)
    return torch.zeros_like(fallback_param)


def safe_cosine(g1, g2, eps=1e-12):
    if g1 is None or g2 is None:
        return np.nan
    a = g1.reshape(-1)
    b = g2.reshape(-1)
    if a.numel() == 0 or b.numel() == 0:
        return np.nan
    if torch.norm(a) <= eps or torch.norm(b) <= eps:
        return np.nan
    return float(F.cosine_similarity(a, b, dim=0, eps=eps).detach().cpu().item())


def compute_norm_ratio_gradients(params, mse_grads, reg_grads, norm_type=2, eps=1e-12, use_global=False):
    scaled_reg_grads = []
    combined_grads = []
    stats_by_param = []

    global_scale = None
    global_mse_norm = None
    global_reg_norm = None

    if use_global:
        ref_device = params[0].device if params else device
        global_mse_norm = global_grad_norm(mse_grads, norm_type=norm_type, device_override=ref_device)
        global_reg_norm = global_grad_norm(reg_grads, norm_type=norm_type, device_override=ref_device)
        if global_reg_norm > global_mse_norm:
            global_scale = global_mse_norm / (global_reg_norm + eps)
            global_scaled = True
        else:
            global_scale = torch.tensor(1.0, device=global_mse_norm.device, dtype=global_mse_norm.dtype)
            global_scaled = False

    for idx, (p, g_mse, g_reg) in enumerate(zip(params, mse_grads, reg_grads)):
        if g_mse is None:
            g_mse = safe_zeros_like(g_reg, p)
        if g_reg is None:
            g_reg = safe_zeros_like(g_mse, p)

        mse_norm = torch.norm(g_mse, p=norm_type)
        reg_norm = torch.norm(g_reg, p=norm_type)

        if use_global:
            scale = global_scale
            scaled = global_scaled
        else:
            if reg_norm > mse_norm:
                scale = mse_norm / (reg_norm + eps)
                scaled = True
            else:
                scale = torch.tensor(1.0, device=g_reg.device, dtype=g_reg.dtype)
                scaled = False

        g_reg_scaled = g_reg * scale
        g_total = g_mse + g_reg_scaled

        scaled_reg_grads.append(g_reg_scaled)
        combined_grads.append(g_total)

        stats_by_param.append(
            {
                "idx": idx,
                "mse_norm": float(mse_norm.detach().cpu().item()),
                "reg_norm": float(reg_norm.detach().cpu().item()),
                "scaled_reg_norm": float(torch.norm(g_reg_scaled, p=norm_type).detach().cpu().item()),
                "combined_norm": float(torch.norm(g_total, p=norm_type).detach().cpu().item()),
                "scale": float(scale.detach().cpu().item()),
                "scaled": bool(scaled),
                "cosine_mse_reg": safe_cosine(g_mse, g_reg, eps=eps),
                "cosine_mse_scaled_reg": safe_cosine(g_mse, g_reg_scaled, eps=eps),
                "cosine_mse_combined": safe_cosine(g_mse, g_total, eps=eps),
            }
        )

    return scaled_reg_grads, combined_grads, stats_by_param


def collect_from_grad_list(model, grads, loss_part_name, run_id, epoch):
    records = []

    for (param_name, _param), grad in zip(model.named_parameters(), grads):
        if grad is None:
            continue

        layer_name = get_linear_layer_name(param_name)
        flat_grad = grad.detach().cpu().numpy().reshape(-1)

        for grad_value in flat_grad:
            records.append(
                {
                    "run_id": int(run_id),
                    "epoch": int(epoch),
                    "loss_part": str(loss_part_name),
                    "layer": str(layer_name),
                    "gradient": float(grad_value),
                }
            )

    return records


def layerwise_cosine_records(model, mse_grads, reg_grads, scaled_reg_grads, combined_grads, stats_by_param, run_id, epoch):
    grouped = {}

    for idx, ((param_name, _param), g_mse, g_reg, g_scaled, g_combined) in enumerate(
        zip(model.named_parameters(), mse_grads, reg_grads, scaled_reg_grads, combined_grads)
    ):
        layer = get_linear_layer_name(param_name)
        if layer not in grouped:
            grouped[layer] = {"mse": [], "reg": [], "scaled_reg": [], "combined": [], "stats": []}

        if g_mse is not None:
            grouped[layer]["mse"].append(g_mse.reshape(-1))
        if g_reg is not None:
            grouped[layer]["reg"].append(g_reg.reshape(-1))
        if g_scaled is not None:
            grouped[layer]["scaled_reg"].append(g_scaled.reshape(-1))
        if g_combined is not None:
            grouped[layer]["combined"].append(g_combined.reshape(-1))
        grouped[layer]["stats"].append(stats_by_param[idx])

    records = []
    for layer, d in grouped.items():
        mse = torch.cat(d["mse"]) if d["mse"] else None
        reg = torch.cat(d["reg"]) if d["reg"] else None
        scaled = torch.cat(d["scaled_reg"]) if d["scaled_reg"] else None
        combined = torch.cat(d["combined"]) if d["combined"] else None
        stats = d["stats"]

        records.append(
            {
                "run_id": int(run_id),
                "epoch": int(epoch),
                "layer": str(layer),
                "mse_norm": float(torch.norm(mse).detach().cpu().item()) if mse is not None else np.nan,
                "reg_norm": float(torch.norm(reg).detach().cpu().item()) if reg is not None else np.nan,
                "scaled_reg_norm": float(torch.norm(scaled).detach().cpu().item()) if scaled is not None else np.nan,
                "combined_norm": float(torch.norm(combined).detach().cpu().item()) if combined is not None else np.nan,
                "reg_scale": float(np.mean([s["scale"] for s in stats])) if stats else np.nan,
                "reg_scaled": int(any(s["scaled"] for s in stats)) if stats else 0,
                "cosine_mse_reg": safe_cosine(mse, reg, eps=grad_eps),
                "cosine_mse_scaled_reg": safe_cosine(mse, scaled, eps=grad_eps),
                "cosine_mse_combined": safe_cosine(mse, combined, eps=grad_eps),
            }
        )

    # Global record over all parameters.
    mse_all = torch.cat([g.reshape(-1) for g in mse_grads if g is not None]) if any(g is not None for g in mse_grads) else None
    reg_all = torch.cat([g.reshape(-1) for g in reg_grads if g is not None]) if any(g is not None for g in reg_grads) else None
    scaled_all = torch.cat([g.reshape(-1) for g in scaled_reg_grads if g is not None]) if any(g is not None for g in scaled_reg_grads) else None
    combined_all = torch.cat([g.reshape(-1) for g in combined_grads if g is not None]) if any(g is not None for g in combined_grads) else None

    records.append(
        {
            "run_id": int(run_id),
            "epoch": int(epoch),
            "layer": "global",
            "mse_norm": float(torch.norm(mse_all).detach().cpu().item()) if mse_all is not None else np.nan,
            "reg_norm": float(torch.norm(reg_all).detach().cpu().item()) if reg_all is not None else np.nan,
            "scaled_reg_norm": float(torch.norm(scaled_all).detach().cpu().item()) if scaled_all is not None else np.nan,
            "combined_norm": float(torch.norm(combined_all).detach().cpu().item()) if combined_all is not None else np.nan,
            "reg_scale": float(np.mean([s["scale"] for s in stats_by_param])) if stats_by_param else np.nan,
            "reg_scaled": int(any(s["scaled"] for s in stats_by_param)) if stats_by_param else 0,
            "cosine_mse_reg": safe_cosine(mse_all, reg_all, eps=grad_eps),
            "cosine_mse_scaled_reg": safe_cosine(mse_all, scaled_all, eps=grad_eps),
            "cosine_mse_combined": safe_cosine(mse_all, combined_all, eps=grad_eps),
        }
    )

    return records


def compute_gradients_for_model(model_path: Path, run_id: int, epoch: int, data_bundle):
    x_data = data_bundle["inputs"]
    y_data = data_bundle["outputs"]
    means = data_bundle["means"]
    stds = data_bundle["stds"]

    model = load_model(
        model_path=model_path,
        input_size=data_bundle["input_size"],
        output_size=data_bundle["output_size"],
    )

    y_pred = model(x_data)
    mse, reg = loss_MSE_norm_ratio(y_pred, y_data, means, stds)

    params, mse_grads, reg_grads = compute_separate_grads(mse, reg, model)
    scaled_reg_grads, combined_grads, stats_by_param = compute_norm_ratio_gradients(
        params,
        mse_grads,
        reg_grads,
        norm_type=2,
        eps=grad_eps,
        use_global=use_global_norm_ratio,
    )

    gradient_records = []
    gradient_records.extend(collect_from_grad_list(model, mse_grads, "mse", run_id, epoch))
    gradient_records.extend(collect_from_grad_list(model, reg_grads, "reg", run_id, epoch))
    gradient_records.extend(collect_from_grad_list(model, scaled_reg_grads, "scaled_reg", run_id, epoch))
    gradient_records.extend(collect_from_grad_list(model, combined_grads, "combined", run_id, epoch))

    cosine_records = layerwise_cosine_records(
        model,
        mse_grads,
        reg_grads,
        scaled_reg_grads,
        combined_grads,
        stats_by_param,
        run_id,
        epoch,
    )

    df_gradients = pd.DataFrame(gradient_records, columns=PLOTTING_COLUMNS)
    if df_gradients.empty:
        df_gradients = pd.DataFrame(columns=PLOTTING_COLUMNS)
    df_gradients = df_gradients[PLOTTING_COLUMNS].copy()
    df_gradients["gradient"] = pd.to_numeric(df_gradients["gradient"], errors="coerce")
    df_gradients = df_gradients.dropna(subset=["run_id", "epoch", "loss_part", "layer", "gradient"])

    df_cosine = pd.DataFrame(cosine_records, columns=COSINE_COLUMNS)
    if df_cosine.empty:
        df_cosine = pd.DataFrame(columns=COSINE_COLUMNS)
    df_cosine = df_cosine[COSINE_COLUMNS].copy()

    return df_gradients, df_cosine


def save_gradients(df: pd.DataFrame, output_csv: Path):
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df = df[PLOTTING_COLUMNS].copy()
    df["gradient"] = pd.to_numeric(df["gradient"], errors="coerce")
    df = df.dropna(subset=["run_id", "epoch", "loss_part", "layer", "gradient"])
    df.to_csv(output_csv, index=False)
    print(f"Saved gradients: {output_csv} | rows={len(df)}")


def save_cosine(df: pd.DataFrame, output_csv: Path):
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df = df[COSINE_COLUMNS].copy()
    df.to_csv(output_csv, index=False)
    print(f"Saved cosine diagnostics: {output_csv} | rows={len(df)}")


def main():
    script_dir = Path(__file__).resolve().parent
    output_root = script_dir / "gradients_by_run"
    cosine_root = script_dir / "cosine_similarity_by_run"

    print(f"Using device: {device}")
    print(f"Looking for models in: {script_dir}")
    print("Offline gradients for norm-ratio adaptive training.")
    print("Stored loss parts: mse, reg, scaled_reg, combined.")
    print("Cosine diagnostics are saved layer-wise and globally.")

    data_bundle = build_dataset()
    print(f"Dataset rows used for offline gradients: {data_bundle['n_rows']}")

    model_files = find_model_files(script_dir)
    if not model_files:
        raise FileNotFoundError(
            "No saved model files were found in the same folder as this script. "
            "Expected exactly one final model named k_model_n.pth or K_model_n.pth, with n between 0 and 4."
        )

    print(f"Model files found: {len(model_files)}")

    failures = []

    for item in model_files:
        run_id = item["run_id"]
        epoch = item["epoch"]
        model_path = item["path"]

        try:
            df_grad, df_cos = compute_gradients_for_model(
                model_path=model_path,
                run_id=run_id,
                epoch=epoch,
                data_bundle=data_bundle,
            )

            grad_csv = output_root / f"run_{run_id}" / f"epoch_{epoch}" / f"gradients_run_{run_id}_epoch_{epoch}.csv"
            cos_csv = cosine_root / f"run_{run_id}" / f"epoch_{epoch}" / f"cosine_run_{run_id}_epoch_{epoch}.csv"

            save_gradients(df_grad, grad_csv)
            save_cosine(df_cos, cos_csv)

        except Exception as e:
            failures.append((model_path.name, str(e)))
            print(f"FAILED model: {model_path.name}")
            print(f"Reason: {e}")

    print(f"Input folder: {script_dir}")
    print(f"Gradient output root: {output_root}")
    print(f"Cosine output root: {cosine_root}")

    if failures:
        print("\nFailures:")
        for name, err in failures:
            print(f" - {name}: {err}")


if __name__ == "__main__":
    main()
