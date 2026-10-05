#!/usr/bin/env python3
"""
Offline raw-gradient calculation for the chemical-kinetics model with

    architecture_e = [300, 300, 300, 300, 300]

Therefore:
    - 5 hidden layers
    - 1 output Linear layer
    - 6 nn.Linear modules total

The script automatically detects exactly one final model:
    k_model_0.pth ... k_model_4.pth
or
    K_model_0.pth ... K_model_4.pth

The detected model number is used as the CV fold index.

It reconstructs Datasets_50 using the same preprocessing as the training:
    - temperature column 2
    - species columns 4:12
    - log(data + 1e-8)
    - global mean/std normalization
    - one-step global shift
    - RepeatedKFold(n_splits=5, n_repeats=1, random_state=42)

It then computes independently:
    L_MSE
    L_mass

and stores the RAW gradients by layer.

Outputs:
    gradients_5_hidden_layers_model_<n>/
        gradient_values_by_layer.csv
        layer_gradient_summary.csv

The first CSV stores every individual gradient element.
The second CSV stores one row per Linear layer with joint weight+bias L2 norms.

No gradient scaling/normalization is applied here.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import RepeatedKFold


# =============================================================================
# USER SETTINGS
# =============================================================================

DATASET_ROOT = Path.home() / "Alternative_trainings" / "Datasets_50"
MODEL_DIR = Path(__file__).resolve().parent

ARCHITECTURE_E = [300, 300, 300, 300, 300]

EXPECTED_HIDDEN_LAYERS = len(ARCHITECTURE_E)      # 5
EXPECTED_LINEAR_LAYERS = EXPECTED_HIDDEN_LAYERS + 1  # 6 including output

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPS = 1e-12

# None = use complete training fold.
MAX_TRAIN_SAMPLES = None
SUBSAMPLE_SEED = 42


# =============================================================================
# MODEL CLASS
# =============================================================================
# Required when a complete model was saved using torch.save(model, ...).

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

        for i, width in enumerate(architecture_e):
            if i == 0:
                layer = nn.Linear(input_size, width)
            else:
                layer = nn.Linear(
                    architecture_e[i - 1],
                    width,
                )

            nn.init.xavier_normal_(layer.weight)
            encoder.append(layer)
            encoder.append(nn.Sigmoid())
            encoder.append(nn.Dropout(p=dropout_p))

        encoder.append(
            nn.Linear(
                architecture_e[-1],
                output_size,
            )
        )

        self.encoder = nn.Sequential(*encoder)

    def forward(self, x):
        return self.encoder(x)


# =============================================================================
# MODEL DISCOVERY
# =============================================================================

def detect_final_model(model_dir: Path):
    pattern = re.compile(
        r"^[kK]_model_([0-4])\.pth$"
    )

    found = []

    for path in model_dir.iterdir():
        if not path.is_file():
            continue

        match = pattern.match(path.name)
        if match:
            found.append(
                (int(match.group(1)), path)
            )

    if len(found) == 0:
        raise FileNotFoundError(
            "\nNo final model was found in:\n"
            f"{model_dir}\n\n"
            "Expected exactly one file such as:\n"
            "    k_model_0.pth\n"
            "    ...\n"
            "    k_model_4.pth\n"
        )

    if len(found) > 1:
        text = "\n".join(
            f"    model {idx}: {path.name}"
            for idx, path in found
        )

        raise RuntimeError(
            "\nMore than one final model was found.\n"
            "Keep only the model you want to analyze in this folder.\n\n"
            f"{text}"
        )

    return found[0]


# =============================================================================
# DATASET
# =============================================================================

def reconstruct_dataset_50(dataset_root: Path):
    sampling_file = (
        dataset_root
        / "T_Sampling_uniform_50.csv"
    )

    if not sampling_file.exists():
        raise FileNotFoundError(
            f"Sampling file not found:\n{sampling_file}"
        )

    sampling = pd.read_csv(sampling_file)

    if "T[K]" in sampling.columns:
        temperatures = sampling["T[K]"].to_numpy()
    else:
        temperatures = sampling.iloc[:, 0].to_numpy()

    trajectories = []

    for T in temperatures:
        filename = (
            "State_space_cte_pressure_"
            f"T{round(float(T), 3)}_phi_1.0.csv"
        )

        path = dataset_root / filename

        if not path.exists():
            raise FileNotFoundError(
                f"State-space file not found:\n{path}"
            )

        df = pd.read_csv(path)

        temperature = (
            df.iloc[:, 2]
            .to_numpy()
            .reshape(-1, 1)
        )

        species = (
            df.iloc[:, 4:12]
            .to_numpy()
        )

        # 1 temperature + 8 species = 9 state variables.
        state = np.concatenate(
            [temperature, species],
            axis=1,
        )

        state_log = np.log(
            state + 1e-8
        )

        trajectories.append(state_log)

    dataset = np.concatenate(
        trajectories,
        axis=0,
    )

    means = dataset.mean(axis=0)
    stds = dataset.std(axis=0)

    stds = np.where(
        stds == 0.0,
        1.0,
        stds,
    )

    normalized = (
        dataset - means
    ) / stds

    inputs = normalized[:-1, :]
    outputs = normalized[1:, :]

    return inputs, outputs, means, stds


def select_training_fold(
    inputs,
    outputs,
    model_index,
):
    splitter = RepeatedKFold(
        n_splits=5,
        n_repeats=1,
        random_state=42,
    )

    for fold_index, (
        train_index,
        test_index,
    ) in enumerate(
        splitter.split(inputs)
    ):
        if fold_index == model_index:
            x_train = inputs[train_index]
            y_train = outputs[train_index]
            return x_train, y_train

    raise ValueError(
        f"Invalid model index: {model_index}"
    )


def optional_subsample(
    x,
    y,
    max_samples,
):
    if (
        max_samples is None
        or len(x) <= max_samples
    ):
        return x, y

    rng = np.random.default_rng(
        SUBSAMPLE_SEED
    )

    idx = rng.choice(
        len(x),
        size=max_samples,
        replace=False,
    )

    return x[idx], y[idx]


# =============================================================================
# MODEL LOADING
# =============================================================================

def load_model(model_path: Path):
    """
    Supports full models saved with torch.save(model, path).
    """

    try:
        model = torch.load(
            model_path,
            map_location=DEVICE,
            weights_only=False,
        )
    except TypeError:
        # Compatibility with older PyTorch.
        model = torch.load(
            model_path,
            map_location=DEVICE,
        )

    if not isinstance(model, nn.Module):
        raise TypeError(
            "\nThe .pth file did not contain a complete nn.Module.\n"
            "This script expects a model saved with torch.save(model, ...)."
        )

    model = model.to(DEVICE)
    model.eval()

    return model


def infer_dtype(model):
    for parameter in model.parameters():
        return parameter.dtype

    return torch.float32


# =============================================================================
# LOSSES
# =============================================================================

def compute_losses(
    model,
    x,
    y,
    means,
    stds,
):
    """
    Reproduce the chemical-kinetics losses.

    MSE:
        mean((prediction - target)^2)

    Mass-conservation loss:
        excludes temperature (column 0)
        transforms normalized log-species back to physical values

        L_mass = mean[
            (sum_i Y_i,true - sum_i Y_i,pred)^2
        ]
    """

    y_pred = model(x)

    mse = torch.mean(
        (y - y_pred) ** 2
    )

    # Species only: columns 1:9.
    pred_log_species = (
        y_pred[:, 1:]
        * stds[1:]
        + means[1:]
    )

    true_log_species = (
        y[:, 1:]
        * stds[1:]
        + means[1:]
    )

    pred_species = torch.exp(
        pred_log_species
    )

    true_species = torch.exp(
        true_log_species
    )

    pred_mass_sum = torch.sum(
        pred_species,
        dim=1,
    )

    true_mass_sum = torch.sum(
        true_species,
        dim=1,
    )

    mass_loss = torch.mean(
        (
            true_mass_sum
            - pred_mass_sum
        ) ** 2
    )

    return mse, mass_loss


# =============================================================================
# GRADIENTS
# =============================================================================

def independent_gradients(
    loss,
    parameters,
    retain_graph,
):
    grads = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=retain_graph,
        create_graph=False,
        allow_unused=True,
    )

    return list(grads)


def get_linear_layer_groups(
    model,
    named_parameters,
):
    """
    Build one group per nn.Linear module.

    With architecture_e = [300,300,300,300,300]:

        Layer 1
        Layer 2
        Layer 3
        Layer 4
        Layer 5
        Output layer

    Weight and bias remain separately identifiable in the detailed CSV,
    while the summary uses their joint L2 norm.
    """

    linear_modules = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear)
    ]

    if len(linear_modules) != EXPECTED_LINEAR_LAYERS:
        raise RuntimeError(
            "\nUnexpected network architecture.\n"
            f"architecture_e is set to: {ARCHITECTURE_E}\n"
            f"Expected {EXPECTED_LINEAR_LAYERS} Linear modules "
            f"({EXPECTED_HIDDEN_LAYERS} hidden + output), "
            f"but loaded model contains {len(linear_modules)}.\n\n"
            "Detected Linear modules:\n"
            + "\n".join(
                f"    {name}: {module}"
                for name, module in linear_modules
            )
        )

    parameter_lookup = {
        name: idx
        for idx, (name, _) in enumerate(
            named_parameters
        )
    }

    groups = []

    for layer_index, (
        module_name,
        module,
    ) in enumerate(
        linear_modules,
        start=1,
    ):
        if layer_index <= EXPECTED_HIDDEN_LAYERS:
            layer_label = f"Layer {layer_index}"
        else:
            layer_label = "Output layer"

        parameter_indices = []

        weight_name = (
            f"{module_name}.weight"
            if module_name
            else "weight"
        )

        bias_name = (
            f"{module_name}.bias"
            if module_name
            else "bias"
        )

        if weight_name in parameter_lookup:
            parameter_indices.append(
                parameter_lookup[weight_name]
            )

        if (
            module.bias is not None
            and bias_name in parameter_lookup
        ):
            parameter_indices.append(
                parameter_lookup[bias_name]
            )

        groups.append({
            "layer_index": layer_index,
            "layer_label": layer_label,
            "module_name": module_name,
            "parameter_indices": parameter_indices,
        })

    return groups


def joint_l2_norm(grads):
    """
    Joint L2 norm of all provided gradient tensors.
    """
    terms = []

    for grad in grads:
        if grad is None:
            continue

        terms.append(
            torch.sum(grad.detach() ** 2)
        )

    if not terms:
        return 0.0

    squared_sum = torch.stack(
        terms
    ).sum()

    return float(
        torch.sqrt(squared_sum)
        .cpu()
        .item()
    )


# =============================================================================
# ANALYSIS
# =============================================================================

def analyze_model(
    model,
    model_index,
    x_np,
    y_np,
    means_np,
    stds_np,
):
    dtype = infer_dtype(model)

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

    means = torch.as_tensor(
        means_np,
        dtype=dtype,
        device=DEVICE,
    )

    stds = torch.as_tensor(
        stds_np,
        dtype=dtype,
        device=DEVICE,
    )

    mse, mass = compute_losses(
        model,
        x,
        y,
        means,
        stds,
    )

    named_parameters = [
        (name, parameter)
        for name, parameter
        in model.named_parameters()
        if parameter.requires_grad
    ]

    parameters = [
        parameter
        for _, parameter
        in named_parameters
    ]

    mse_grads = independent_gradients(
        mse,
        parameters,
        retain_graph=True,
    )

    mass_grads = independent_gradients(
        mass,
        parameters,
        retain_graph=False,
    )

    groups = get_linear_layer_groups(
        model,
        named_parameters,
    )

    # -------------------------------------------------------------------------
    # Individual gradient elements
    # -------------------------------------------------------------------------

    gradient_rows = []

    parameter_to_group = {}

    for group in groups:
        for parameter_index in group[
            "parameter_indices"
        ]:
            parameter_to_group[
                parameter_index
            ] = group

    for parameter_index, (
        parameter_name,
        parameter,
    ) in enumerate(
        named_parameters
    ):
        if parameter_index not in parameter_to_group:
            continue

        group = parameter_to_group[
            parameter_index
        ]

        if parameter_name.endswith(
            ".weight"
        ):
            parameter_type = "weight"
        elif parameter_name.endswith(
            ".bias"
        ):
            parameter_type = "bias"
        else:
            parameter_type = "parameter"

        for loss_part, grad in [
            ("mse", mse_grads[parameter_index]),
            ("mass", mass_grads[parameter_index]),
        ]:
            if grad is None:
                continue

            flat = (
                grad.detach()
                .cpu()
                .numpy()
                .reshape(-1)
            )

            for value in flat:
                gradient_rows.append({
                    "model_index": model_index,
                    "layer_index": group[
                        "layer_index"
                    ],
                    "layer_label": group[
                        "layer_label"
                    ],
                    "module_name": group[
                        "module_name"
                    ],
                    "parameter_name": parameter_name,
                    "parameter_type": parameter_type,
                    "loss_part": loss_part,
                    "gradient": float(value),
                })

    # -------------------------------------------------------------------------
    # Layer-wise summary
    # -------------------------------------------------------------------------

    summary_rows = []

    for group in groups:
        idxs = group[
            "parameter_indices"
        ]

        mse_layer_grads = [
            mse_grads[i]
            for i in idxs
        ]

        mass_layer_grads = [
            mass_grads[i]
            for i in idxs
        ]

        mse_norm = joint_l2_norm(
            mse_layer_grads
        )

        mass_norm = joint_l2_norm(
            mass_layer_grads
        )

        R_l = (
            mass_norm
            / (mse_norm + EPS)
        )

        log10_R_l = np.log10(
            max(
                R_l,
                np.finfo(float).tiny,
            )
        )

        n_mse = sum(
            int(g.numel())
            for g in mse_layer_grads
            if g is not None
        )

        n_mass = sum(
            int(g.numel())
            for g in mass_layer_grads
            if g is not None
        )

        summary_rows.append({
            "model_index": model_index,
            "layer_index": group[
                "layer_index"
            ],
            "layer_label": group[
                "layer_label"
            ],
            "module_name": group[
                "module_name"
            ],
            "mse_loss": float(
                mse.detach().cpu().item()
            ),
            "mass_loss": float(
                mass.detach().cpu().item()
            ),
            "mse_grad_norm": mse_norm,
            "mass_grad_norm": mass_norm,
            "R_l_raw": R_l,
            "log10_R_l_raw": log10_R_l,
            "n_mse_elements": n_mse,
            "n_mass_elements": n_mass,
        })

    return (
        pd.DataFrame(gradient_rows),
        pd.DataFrame(summary_rows),
    )


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 78)
    print("5-HIDDEN-LAYER CHEMICAL-KINETICS GRADIENT CALCULATION")
    print("=" * 78)

    print(
        f"architecture_e       : {ARCHITECTURE_E}"
    )
    print(
        f"expected hidden layers: {EXPECTED_HIDDEN_LAYERS}"
    )
    print(
        f"expected Linear layers: {EXPECTED_LINEAR_LAYERS}"
    )
    print(
        f"device               : {DEVICE}"
    )
    print()

    model_index, model_path = (
        detect_final_model(
            MODEL_DIR
        )
    )

    print(
        f"Detected model index: {model_index}"
    )
    print(
        f"Model file          : {model_path}"
    )

    print("\nReconstructing Datasets_50...")

    (
        inputs,
        outputs,
        means,
        stds,
    ) = reconstruct_dataset_50(
        DATASET_ROOT
    )

    x_train, y_train = (
        select_training_fold(
            inputs,
            outputs,
            model_index,
        )
    )

    x_train, y_train = (
        optional_subsample(
            x_train,
            y_train,
            MAX_TRAIN_SAMPLES,
        )
    )

    print(
        f"Training samples used: {len(x_train)}"
    )

    print("\nLoading model...")

    model = load_model(
        model_path
    )

    linear_modules = [
        (name, module)
        for name, module
        in model.named_modules()
        if isinstance(module, nn.Linear)
    ]

    print(
        f"Linear modules detected: {len(linear_modules)}"
    )

    for i, (name, module) in enumerate(
        linear_modules,
        start=1,
    ):
        label = (
            f"Layer {i}"
            if i <= EXPECTED_HIDDEN_LAYERS
            else "Output layer"
        )

        print(
            f"  {label:<12} "
            f"{name:<20} "
            f"{module.in_features} -> {module.out_features}"
        )

    print("\nCalculating raw MSE and mass gradients...")

    gradient_df, summary_df = (
        analyze_model(
            model,
            model_index,
            x_train,
            y_train,
            means,
            stds,
        )
    )

    output_dir = (
        MODEL_DIR
        / f"gradients_5_hidden_layers_model_{model_index}"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    gradient_csv = (
        output_dir
        / "gradient_values_by_layer.csv"
    )

    summary_csv = (
        output_dir
        / "layer_gradient_summary.csv"
    )

    gradient_df.to_csv(
        gradient_csv,
        index=False,
    )

    summary_df.to_csv(
        summary_csv,
        index=False,
    )

    print("\nLayer gradient summary:")
    print(
        summary_df[
            [
                "layer_index",
                "layer_label",
                "mse_grad_norm",
                "mass_grad_norm",
                "R_l_raw",
                "log10_R_l_raw",
            ]
        ].to_string(
            index=False
        )
    )

    print("\nSaved:")
    print(
        f"  {gradient_csv}"
    )
    print(
        f"  {summary_csv}"
    )

    print(
        "\nThe detailed CSV contains every individual MSE and "
        "mass-conservation gradient element, identified by layer, "
        "weight/bias, and parameter name."
    )


if __name__ == "__main__":
    main()
