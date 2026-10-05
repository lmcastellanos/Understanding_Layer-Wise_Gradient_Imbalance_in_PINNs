import re
from pathlib import Path
from os.path import join
from os import environ

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

home = environ["HOME"]
folder1 = "Alternative_trainings"
folder2 = "Datasets_40"

phi_name = 1.0
n_autoregression = 1
REG_WEIGHT = 1e-12

name_11 = "State_space_cte_pressure_T"
name_12 = "_phi_"
end = ".csv"

PLOTTING_COLUMNS = ["run_id", "epoch", "loss_part", "layer", "gradient"]


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
    condition_csv = join(home, folder1, folder2, "T_Sampling_uniform_40.csv")
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
                    nn.init.xavier_normal_(layer.weight)
                    encoder.append(layer)
                    encoder.append(nn.Sigmoid())
                    encoder.append(nn.Dropout(p=dropout_p))
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


def loss_MSE_with_reg(y_pred, y_train, means, stds):
    mse = torch.mean(torch.square(y_train - y_pred))

    # Build the regularizer fully in torch so its gradient is preserved.
    means = means.to(device=y_pred.device, dtype=y_pred.dtype)
    stds = stds.to(device=y_pred.device, dtype=y_pred.dtype)

    pred_mass = y_pred[:, 1:] * stds[1:] + means[1:]
    true_mass = y_train[:, 1:] * stds[1:] + means[1:]

    pred_mass = torch.exp(pred_mass)
    true_mass = torch.exp(true_mass)

    reg = torch.square(torch.sum(true_mass) - torch.sum(pred_mass)).mean()
    reg = REG_WEIGHT * reg
    reg = reg.to(dtype=mse.dtype)

    return mse, reg


def parse_model_file_info(path: Path):
    name = path.name

    m_epoch = re.match(r"k_model_epochs_(\d+)_(\d+)\.pth$", name)
    if m_epoch:
        return {"epoch": int(m_epoch.group(1)), "run_id": int(m_epoch.group(2)), "kind": "checkpoint"}

    m_final = re.match(r"k_model_(\d+)\.pth$", name)
    if m_final:
        return {"epoch": None, "run_id": int(m_final.group(1)), "kind": "final"}

    return None


def infer_final_epoch_from_loss_csv(script_dir: Path, run_id: int):
    candidates = [
        script_dir / f"k_model_loss{run_id}.csv",
        script_dir / f"k_model_reg_loss{run_id}.csv",
    ]

    lengths = []
    for csv_path in candidates:
        if not csv_path.exists():
            continue
        try:
            arr = np.loadtxt(csv_path, delimiter=",")
            arr = np.atleast_1d(arr)
            lengths.append(int(len(arr)))
        except Exception:
            continue

    if lengths:
        return max(lengths)
    return None


def find_model_files(script_dir: Path):
    model_files = []
    for path in sorted(script_dir.glob("*.pth")):
        info = parse_model_file_info(path)
        if info is None:
            continue

        if info["epoch"] is None:
            info["epoch"] = infer_final_epoch_from_loss_csv(script_dir, info["run_id"])

        if info["epoch"] is None:
            continue

        info["path"] = path
        model_files.append(info)

    return model_files


def get_linear_layer_name(param_name: str):
    m = re.match(r"encoder\.(\d+)\.(weight|bias)$", param_name)
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


def collect_gradients(model, loss_tensor, loss_part_name, run_id, epoch):
    if not isinstance(loss_tensor, torch.Tensor) or not loss_tensor.requires_grad:
        return []

    model.zero_grad(set_to_none=True)
    loss_tensor.backward()

    records = []
    for param_name, param in model.named_parameters():
        if param.grad is None:
            continue

        layer_name = get_linear_layer_name(param_name)
        flat_grad = param.grad.detach().cpu().numpy().reshape(-1)

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


def compute_gradients_for_model(model_path: Path, run_id: int, epoch: int, data_bundle, script_dir: Path):
    x_data = data_bundle["inputs"]
    y_data = data_bundle["outputs"]
    means = data_bundle["means"]
    stds = data_bundle["stds"]

    model = load_model(
        model_path=model_path,
        input_size=data_bundle["input_size"],
        output_size=data_bundle["output_size"],
    )

    def build_losses():
        y_pred = model(x_data)
        mse, reg = loss_MSE_with_reg(y_pred, y_data, means, stds)
        total_loss = mse + reg
        return mse, reg, total_loss

    records = []

    mse, _, _ = build_losses()
    records.extend(collect_gradients(model, mse, "mse", run_id, epoch))

    _, reg, _ = build_losses()
    records.extend(collect_gradients(model, reg, "reg", run_id, epoch))

    _, _, total_loss = build_losses()
    records.extend(collect_gradients(model, total_loss, "total", run_id, epoch))

    df = pd.DataFrame(records, columns=PLOTTING_COLUMNS)
    if df.empty:
        df = pd.DataFrame(columns=PLOTTING_COLUMNS)

    df = df[PLOTTING_COLUMNS].copy()
    df["gradient"] = pd.to_numeric(df["gradient"], errors="coerce")
    df = df.dropna(subset=["run_id", "epoch", "loss_part", "layer", "gradient"])

    return df


def save_gradients(df: pd.DataFrame, output_csv: Path):
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df = df[PLOTTING_COLUMNS].copy()
    df["gradient"] = pd.to_numeric(df["gradient"], errors="coerce")
    df = df.dropna(subset=["run_id", "epoch", "loss_part", "layer", "gradient"])
    df.to_csv(output_csv, index=False)
    print(f"Saved compatible gradients: {output_csv} | rows={len(df)}")


def main():
    script_dir = Path(__file__).resolve().parent
    output_root = script_dir / "gradients_by_run"

    print(f"Using device: {device}")
    print(f"Looking for models in: {script_dir}")
    print(f"Using regularization constant: {REG_WEIGHT}")
    print("The reg term is rebuilt in torch so mse, reg, and total gradients are all stored.")

    data_bundle = build_dataset()
    print(f"Dataset rows used for offline gradients: {data_bundle['n_rows']}")

    model_files = find_model_files(script_dir)
    if not model_files:
        raise FileNotFoundError(
            "No saved model files were found in the same folder as this script. "
            "Expected names like k_model_epochs_1000_0.pth or k_model_0.pth."
        )

    print(f"Model files found: {len(model_files)}")

    failures = []

    for item in model_files:
        run_id = item["run_id"]
        epoch = item["epoch"]
        model_path = item["path"]

        try:
            df_grad = compute_gradients_for_model(
                model_path=model_path,
                run_id=run_id,
                epoch=epoch,
                data_bundle=data_bundle,
                script_dir=script_dir,
            )

            output_csv = output_root / f"run_{run_id}" / f"epoch_{epoch}" / f"gradients_run_{run_id}_epoch_{epoch}.csv"
            save_gradients(df_grad, output_csv)

        except Exception as e:
            failures.append((model_path.name, str(e)))
            print(f"FAILED model: {model_path.name}")
            print(f"Reason: {e}")

    print(f"Input folder: {script_dir}")
    print(f"Output root: {output_root}")

    if failures:
        print("\nFailures:")
        for name, err in failures:
            print(f" - {name}: {err}")


if __name__ == "__main__":
    main()
