import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from numpy.linalg import norm
from sklearn.metrics import r2_score
from scipy.spatial.distance import directed_hausdorff


# -----------------------------------------------------------------------------
# Publication-style formatting
# -----------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 300,
    "lines.linewidth": 1.5,
    "savefig.dpi": 600,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# -----------------------------------------------------------------------------
# Paths and settings
# -----------------------------------------------------------------------------
HOME = Path.home()
DATASET_ROOT = HOME / "Alternative_trainings" / "Datasets_30"
EVALUATION_CASES = ["interpolation", "extrapolation"]

STATE_PREFIX = "State_space_cte_pressure_T"
STATE_PHI_TOKEN = "_phi_"

N_AUTOREGRESSION = 1

# Variables to evaluate/plot. Their order is preserved in the output CSV files.
INTEREST_VECTOR = ["H2O", "O2", "H2", "T[K]", "OH", "HO2", "H2O2", "O", "H"]

# Folder in which K_model_0.pth or K_model_1.pth is expected.
MODEL_FOLDER = Path(".")
OUTPUT_ROOT = Path("model_evaluation")


# -----------------------------------------------------------------------------
# Data preprocessing
# -----------------------------------------------------------------------------
def hydrogen_log_transform_sandiego_cantera(cantera_species):
    cantera_time = cantera_species.iloc[:, 1]
    cantera_temperature = cantera_species.iloc[:, 2]
    cantera_species_fractions = cantera_species.iloc[:, 4:12]

    n_columns_mass_fraction = cantera_species_fractions.shape[1]

    data_all = pd.concat(
        [cantera_temperature, cantera_species_fractions],
        axis=1,
    )
    data_all = np.log(data_all + 1e-8)

    columns = data_all.columns.to_list()

    return data_all, n_columns_mass_fraction, columns, cantera_time


def z_score_data_transformation_time_shift(
    dataset,
    n_autoregression,
    columns,
    means,
    stds,
):
    """Normalize with training statistics and construct shifted state pairs."""
    dataset = dataset.copy()

    means_np = means.to_numpy()
    stds_np = stds.to_numpy()

    for j in range(dataset.shape[1]):
        mean = means_np[0, j + 1]
        std = stds_np[0, j + 1]
        dataset.iloc[:, j] = (dataset.iloc[:, j] - mean) / std

    if n_autoregression != 1:
        # Kept for compatibility with the original workflow.
        dataset_copy = dataset.copy()
        core_data = dataset_copy.iloc[1:-n_autoregression, 1:].reset_index(drop=True)
        shifted_dfs = [core_data]

        for i in range(1, n_autoregression + 2):
            shifted_df = dataset.copy().iloc[i:-n_autoregression + 1, :]
            shifted_df = shifted_df.reset_index(drop=True)
            shifted_dfs.append(shifted_df)

        result_df = pd.concat(shifted_dfs, axis=1)
        dataset_out = result_df.iloc[:-n_autoregression, :]
    else:
        dataset_t0 = dataset.iloc[:-1, :].reset_index(drop=True)
        dataset_shift = dataset.iloc[1:, :].reset_index(drop=True)
        dataset_shift = dataset_shift.add_suffix("_shift")
        dataset_out = pd.concat([dataset_t0, dataset_shift], axis=1)

    return dataset_out


# -----------------------------------------------------------------------------
# Network definition
# Must be present because the original models were saved with torch.save(model).
# -----------------------------------------------------------------------------
class NN(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        dropout_p,
        architecture_e,
        min_val,
        max_val,
        alpha,
    ):
        super(NN, self).__init__()

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


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------
def curve_matching(y_true, y_pred, time_plot):
    """Original curve-matching metric retained from the supplied script."""
    width = len(time_plot)
    grid = np.arange(len(time_plot))

    dy_true = np.gradient(y_true, grid)
    dy_pred = np.gradient(y_pred, grid)

    max_true = np.max(np.abs(y_true))
    if max_true == 0:
        max_true = 1.0

    max_dy_true = np.max(np.abs(dy_true))
    if max_dy_true == 0:
        max_dy_true = 1.0

    f0 = y_true / max_true
    g0 = y_pred / max_true
    f1 = dy_true / max_dy_true
    g1 = dy_pred / max_dy_true

    d0l2 = 1 / (1 + norm(f0 - g0) / width)
    d1l2 = 1 / (1 + norm(f1 - g1) / width)

    norm_f0 = norm(f0)
    norm_g0 = norm(g0)
    norm_f1 = norm(f1)
    norm_g1 = norm(g1)

    d0p = np.nan if norm_f0 == 0 or norm_g0 == 0 else 1 - 0.5 * norm(f0 / norm_f0 - g0 / norm_g0)
    d1p = np.nan if norm_f1 == 0 or norm_g1 == 0 else 1 - 0.5 * norm(f1 / norm_f1 - g1 / norm_g1)

    M = np.nanmean([d0l2, d1l2, d0p, d1p])
    return np.array([d0l2, d1l2, d0p, d1p, M])


def hausdorff_curve_distance(time_plot, y_true, y_pred):
    """
    Symmetric Hausdorff distance between the true and reconstructed curves.

    Both time and ordinate are normalized before computing the distance so that
    the metric is not dominated by units or by the scale of a particular species.
    The same normalization, derived from the reference curve, is applied to the
    reconstruction. Lower values are better; 0 means coincident point sets.
    """
    t = np.asarray(time_plot, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    t_range = np.ptp(t)
    if t_range == 0:
        t_range = 1.0
    t_norm = (t - np.min(t)) / t_range

    y_min = np.min(y_true)
    y_range = np.ptp(y_true)
    if y_range == 0:
        y_range = 1.0

    y_true_norm = (y_true - y_min) / y_range
    y_pred_norm = (y_pred - y_min) / y_range

    true_curve = np.column_stack((t_norm, y_true_norm))
    pred_curve = np.column_stack((t_norm, y_pred_norm))

    d_true_pred = directed_hausdorff(true_curve, pred_curve)[0]
    d_pred_true = directed_hausdorff(pred_curve, true_curve)[0]

    return max(d_true_pred, d_pred_true)


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------
def find_model_files(folder):
    """Find K_model_n.pth automatically; n is expected to be 0 or 1."""
    candidates = list(folder.glob("K_model_*.pth")) + list(folder.glob("k_model_*.pth"))

    valid_models = []
    for path in candidates:
        match = re.fullmatch(r"[Kk]_model_(\d+)\.pth", path.name)
        if match is None:
            continue

        model_number = int(match.group(1))
        if model_number in (0, 1, 2, 3, 4):
            valid_models.append((model_number, path))

    # Remove duplicates and sort by the detected model number.
    unique = {}
    for model_number, path in valid_models:
        unique[path.resolve()] = (model_number, path)

    models = sorted(unique.values(), key=lambda item: item[0])

    if not models:
        raise FileNotFoundError(
            f"No K_model_0.pth or K_model_1.pth found in {folder.resolve()}"
        )

    return models


def find_dataset_files(case_folder):
    """Find every State_space CSV in interpolation/extrapolation automatically."""
    files = sorted(case_folder.glob(f"{STATE_PREFIX}*{STATE_PHI_TOKEN}*.csv"))

    if not files:
        raise FileNotFoundError(
            f"No state-space CSV files found in {case_folder.resolve()}"
        )

    return files


def safe_name(text):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)


def load_full_model(model_path):
    """Load full-model checkpoints across older and newer PyTorch versions."""
    try:
        # PyTorch >= 2.6 defaults weights_only=True; these checkpoints are full models.
        return torch.load(
            model_path,
            map_location=torch.device("cpu"),
            weights_only=False,
        )
    except TypeError:
        # Older PyTorch versions do not expose weights_only.
        return torch.load(model_path, map_location=torch.device("cpu"))


def inverse_transform_column(values, results_index, means, stds):
    values = np.asarray(values)
    values = values * stds.iloc[0, results_index + 1] + means.iloc[0, results_index + 1]
    return np.exp(values)


def save_metric_tables(output_folder, mode_name, case_names, r2_values, hausdorff_values, quality_values):
    r2_df = pd.DataFrame(r2_values, index=case_names, columns=INTEREST_VECTOR)
    r2_df.index.name = "case"
    r2_df.to_csv(output_folder / f"{mode_name}_r2_score.csv")

    hausdorff_df = pd.DataFrame(hausdorff_values, index=case_names, columns=INTEREST_VECTOR)
    hausdorff_df.index.name = "case"
    hausdorff_df.to_csv(output_folder / f"{mode_name}_hausdorff_distance.csv")

    quality_df = pd.DataFrame(quality_values, index=case_names, columns=INTEREST_VECTOR)
    quality_df.index.name = "case"
    quality_df.to_csv(output_folder / f"{mode_name}_qualities.csv")


# -----------------------------------------------------------------------------
# Evaluation of one interpolation/extrapolation branch
# -----------------------------------------------------------------------------
def evaluate_case_folder(network, case_name, case_folder, output_folder, means, stds):
    dataset_files = find_dataset_files(case_folder)

    element_folder = output_folder / "element_wise"
    autoreg_folder = output_folder / "autoregression"
    element_folder.mkdir(parents=True, exist_ok=True)
    autoreg_folder.mkdir(parents=True, exist_ok=True)

    n_cases = len(dataset_files)
    n_variables = len(INTEREST_VECTOR)

    element_r2 = np.full((n_cases, n_variables), np.nan)
    element_hd = np.full((n_cases, n_variables), np.nan)
    element_quality = np.full((n_cases, n_variables), np.nan)

    autoreg_r2 = np.full((n_cases, n_variables), np.nan)
    autoreg_hd = np.full((n_cases, n_variables), np.nan)
    autoreg_quality = np.full((n_cases, n_variables), np.nan)

    case_labels = []

    for i, csv_path in enumerate(dataset_files):
        print(f"    [{i + 1}/{n_cases}] {csv_path.name}")

        cantera_species = pd.read_csv(csv_path)
        cantera_species = pd.DataFrame(cantera_species)

        dataset_raw, n_columns_mass_fraction, columns, time = hydrogen_log_transform_sandiego_cantera(
            cantera_species
        )
        dataset = z_score_data_transformation_time_shift(
            dataset_raw,
            N_AUTOREGRESSION,
            columns,
            means,
            stds,
        )

        state_size = 1 + n_columns_mass_fraction
        inputs = dataset.iloc[:, :state_size].to_numpy(dtype=np.float64)
        targets = dataset.iloc[:, state_size:2 * state_size].to_numpy(dtype=np.float64)

        time_plot = time.iloc[1:].to_numpy(dtype=float) * 1e3

        # The target half is the true next state corresponding to each network output.
        target_df = dataset.iloc[:, state_size:2 * state_size]

        # ---------------------------------------------------------------------
        # Element-wise evaluation: each true x_t is independently mapped to x_(t+1)
        # ---------------------------------------------------------------------
        with torch.no_grad():
            input_tensor = torch.from_numpy(inputs)
            element_results = network(input_tensor).cpu().numpy()

        # ---------------------------------------------------------------------
        # Autoregressive evaluation: only x_0 is supplied; predictions are fed back
        # ---------------------------------------------------------------------
        autoreg_results = np.zeros_like(targets)
        input_it = torch.from_numpy(inputs[0])

        with torch.no_grad():
            for k in range(len(inputs)):
                result = network(input_it)
                autoreg_results[k, :] = result.cpu().numpy()
                input_it = result

        case_label = csv_path.stem
        case_labels.append(case_label)
        plot_case_name = safe_name(case_label)

        for l, variable in enumerate(INTEREST_VECTOR):
            if variable not in columns:
                print(f"      Warning: {variable} not found in columns; skipping")
                continue

            results_index = columns.index(variable)

            # Correct next-state reference for both evaluation modes.
            original_scaled = target_df.iloc[:, results_index].to_numpy()
            original = inverse_transform_column(original_scaled, results_index, means, stds)

            element_output = inverse_transform_column(
                element_results[:, results_index],
                results_index,
                means,
                stds,
            )

            autoreg_output = inverse_transform_column(
                autoreg_results[:, results_index],
                results_index,
                means,
                stds,
            )

            # ------------------------- Element-wise metrics ------------------
            element_r2[i, l] = r2_score(original, element_output)
            element_hd[i, l] = hausdorff_curve_distance(time_plot, original, element_output)
            element_quality[i, l] = curve_matching(original, element_output, time_plot)[-1]

            # ------------------------- Autoregressive metrics -----------------
            autoreg_r2[i, l] = r2_score(original, autoreg_output)
            autoreg_hd[i, l] = hausdorff_curve_distance(time_plot, original, autoreg_output)
            autoreg_quality[i, l] = curve_matching(original, autoreg_output, time_plot)[-1]

            # ------------------------- Element-wise plot ----------------------
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.scatter(time_plot, original, label=f"{variable} Dataset")
            ax.scatter(time_plot, element_output, label=f"{variable} Reconstruction")
            ax.set_xlabel("Time [mS]", fontsize=30)
            ax.set_ylabel(variable, fontsize=30, labelpad=40, rotation="horizontal")
            ax.tick_params(axis="both", labelsize=30)
            ax.legend(fontsize=24)
            ax.set_title(
                f"{case_name.capitalize()} | Element-wise | R²={element_r2[i, l]:.5f} | "
                f"Hausdorff={element_hd[i, l]:.5f}",
                fontsize=16,
            )
            fig.savefig(
                element_folder / f"element_wise_{safe_name(variable)}_{plot_case_name}.png",
                dpi=600,
                bbox_inches="tight",
            )
            plt.close(fig)

            # ------------------------- Autoregressive plot --------------------
            fig, ax = plt.subplots(figsize=(13, 8))
            ax.scatter(time_plot, original, label=f"{variable} Dataset")
            ax.scatter(time_plot, autoreg_output, label=f"{variable} Reconstruction")
            ax.set_xlabel("Time [mS]", fontsize=30)
            ax.set_ylabel(variable, fontsize=30, labelpad=40, rotation="horizontal")
            ax.tick_params(axis="both", labelsize=30)
            ax.legend(fontsize=24)
            ax.set_title(
                f"{case_name.capitalize()} | Autoregression | R²={autoreg_r2[i, l]:.5f} | "
                f"Hausdorff={autoreg_hd[i, l]:.5f}",
                fontsize=16,
            )
            fig.savefig(
                autoreg_folder / f"autoregression_{safe_name(variable)}_{plot_case_name}.png",
                dpi=600,
                bbox_inches="tight",
            )
            plt.close(fig)

    # Save one CSV per metric and prediction mode.
    save_metric_tables(
        element_folder,
        "element_wise",
        case_labels,
        element_r2,
        element_hd,
        element_quality,
    )

    save_metric_tables(
        autoreg_folder,
        "autoregression",
        case_labels,
        autoreg_r2,
        autoreg_hd,
        autoreg_quality,
    )

    # Also save compact summary tables with averages over all curves/species.
    summary = pd.DataFrame({
        "mode": ["element_wise", "autoregression"],
        "mean_r2": [np.nanmean(element_r2), np.nanmean(autoreg_r2)],
        "mean_hausdorff": [np.nanmean(element_hd), np.nanmean(autoreg_hd)],
        "mean_curve_quality": [np.nanmean(element_quality), np.nanmean(autoreg_quality)],
    })
    summary.to_csv(output_folder / "metrics_summary.csv", index=False)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    means_path = DATASET_ROOT / "mean_values_log.csv"
    stds_path = DATASET_ROOT / "std_values_log.csv"

    if not means_path.exists():
        raise FileNotFoundError(f"Missing normalization file: {means_path}")
    if not stds_path.exists():
        raise FileNotFoundError(f"Missing normalization file: {stds_path}")

    means = pd.read_csv(means_path)
    stds = pd.read_csv(stds_path)

    models = find_model_files(MODEL_FOLDER)

    print("Models found:")
    for model_number, model_path in models:
        print(f"  model {model_number}: {model_path.resolve()}")

    for model_number, model_path in models:
        print(f"\nLoading {model_path.name}")

        network = load_full_model(model_path)
        network = network.double()
        network.eval()

        # If there is only one model, keep the requested clean directory structure.
        # If two models are present, use separate directories to avoid overwriting
        # while still keeping model numbers out of individual output filenames.
        if len(models) == 1:
            model_output_root = OUTPUT_ROOT
        else:
            model_output_root = OUTPUT_ROOT / f"model_{model_number}"

        for case_name in EVALUATION_CASES:
            case_folder = DATASET_ROOT / case_name

            if not case_folder.exists():
                print(f"Skipping missing folder: {case_folder}")
                continue

            print(f"  Evaluating {case_name}: {case_folder}")

            output_folder = model_output_root / case_name
            output_folder.mkdir(parents=True, exist_ok=True)

            evaluate_case_folder(
                network=network,
                case_name=case_name,
                case_folder=case_folder,
                output_folder=output_folder,
                means=means,
                stds=stds,
            )

    print("\nEvaluation completed.")
    print(f"Results saved under: {OUTPUT_ROOT.resolve()}")


if __name__ == "__main__":
    main()
