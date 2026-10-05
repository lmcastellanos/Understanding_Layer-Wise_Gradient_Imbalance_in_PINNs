#!/usr/bin/env python3

"""
Evaluate ignition delay time (IDT) in Datasets_50 interpolation cases.

For each interpolation case, this script computes:

REFERENCE
---------
real_IDT_ms:
    Reference IDT defined from the physical temperature trajectory as the
    time of maximum dT/dt.

PREDICTIONS
-----------
For BOTH element-wise and autoregressive reconstruction:
    temperature_IDT_ms:
        time of maximum dT_pred/dt

    temperature_IDT_difference_ms:
        |temperature_IDT_ms - real_IDT_ms|

    OH_IDT_ms:
        time of maximum dOH_pred/dt

    OH_IDT_difference_ms:
        |OH_IDT_ms - real_IDT_ms|

Outputs
-------
model_evaluation_IDT_Datasets_50/
    element_wise_IDT_comparison.csv
    autoregression_IDT_comparison.csv
"""

import re
from pathlib import Path
from os import environ

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# ============================================================
# USER SETTINGS
# ============================================================

DATASET_FOLDER = "Datasets_50"
EVALUATION_CASE = "interpolation"

STATE_PREFIX = "State_space_cte_pressure_T"
STATE_SUFFIX = "_phi_"

OUTPUT_FOLDER = "model_evaluation_IDT_Datasets_50"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# MODEL DEFINITION
# ============================================================

class NN(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        dropout_p,
        architecture_e,
        min_val=None,
        max_val=None,
        alpha=0.001,
    ):
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
                    layer = nn.Linear(
                        architecture_e[i - 1],
                        architecture_e[i],
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


# ============================================================
# MODEL DETECTION / LOADING
# ============================================================

def detect_model(script_dir: Path):
    """
    Detect exactly one final model:
        k_model_0.pth ... k_model_4.pth
        K_model_0.pth ... K_model_4.pth
    """

    pattern = re.compile(
        r"^[Kk]_model_([0-4])\.pth$"
    )

    candidates = []

    for path in sorted(
        script_dir.rglob("*.pth")
    ):
        match = pattern.match(
            path.name
        )

        if match:
            candidates.append(
                (
                    int(match.group(1)),
                    path,
                )
            )

    if len(candidates) == 0:
        raise FileNotFoundError(
            "No final model found. Expected exactly one "
            "k_model_n.pth or K_model_n.pth with n between 0 and 4."
        )

    if len(candidates) > 1:
        raise RuntimeError(
            "More than one final model was found: "
            + ", ".join(
                str(path)
                for _, path in candidates
            )
        )

    model_number, model_path = candidates[0]

    print(
        f"Detected model: {model_path.name} "
        f"(model number = {model_number})"
    )

    return model_number, model_path


def rebuild_model_from_state_dict(
    state_dict,
    input_size,
    output_size,
):
    model = NN(
        input_size=input_size,
        output_size=output_size,
        dropout_p=0.0,
        architecture_e=[300, 300, 300],
        min_val=None,
        max_val=None,
    )

    cleaned = {}

    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module."):]
        cleaned[key] = value

    model.load_state_dict(
        cleaned,
        strict=True,
    )

    model = model.to(DEVICE)
    model.eval()

    return model


def load_model(
    model_path,
    input_size,
    output_size,
):
    try:
        obj = torch.load(
            model_path,
            map_location=DEVICE,
        )
    except Exception:
        obj = torch.load(
            model_path,
            map_location=DEVICE,
            weights_only=False,
        )

    if isinstance(obj, nn.DataParallel):
        obj = obj.module

    if isinstance(obj, nn.Module):
        obj = obj.to(DEVICE)
        obj.eval()
        return obj

    if isinstance(obj, dict):
        return rebuild_model_from_state_dict(
            obj,
            input_size,
            output_size,
        )

    raise TypeError(
        f"Unsupported model content in {model_path}"
    )


# ============================================================
# DATA PREPARATION
# ============================================================

def hydrogen_log_transform(
    cantera_species,
):
    """
    Match the training/evaluation convention:
      - temperature = column 2
      - species mass fractions = columns 4:12
    """

    time = cantera_species.iloc[:, 1]
    temperature = cantera_species.iloc[:, 2]
    species_fractions = cantera_species.iloc[:, 4:12]

    data_all = pd.concat(
        [
            temperature,
            species_fractions,
        ],
        axis=1,
    )

    data_all = np.log(
        data_all + 1e-8
    )

    columns = data_all.columns.to_list()

    return (
        data_all,
        columns,
        time,
    )


def z_score_and_shift(
    dataset,
    means,
    stds,
):
    """
    Normalize with the stored training statistics and build
    current-state / next-state pairs.
    """

    dataset = dataset.copy()

    mean_values = means.to_numpy()
    std_values = stds.to_numpy()

    for j in range(
        dataset.shape[1]
    ):
        mean = mean_values[0, j + 1]
        std = std_values[0, j + 1]

        dataset.iloc[:, j] = (
            dataset.iloc[:, j]
            - mean
        ) / std

    current = (
        dataset.iloc[:-1, :]
        .reset_index(drop=True)
    )

    shifted = (
        dataset.iloc[1:, :]
        .reset_index(drop=True)
        .add_suffix("_shift")
    )

    return pd.concat(
        [
            current,
            shifted,
        ],
        axis=1,
    )


def inverse_transform_column(
    normalized_values,
    column_index,
    means,
    stds,
):
    values = (
        np.asarray(
            normalized_values,
            dtype=float,
        )
        * stds.iloc[
            0,
            column_index + 1,
        ]
        + means.iloc[
            0,
            column_index + 1,
        ]
    )

    return np.exp(
        values
    )


def find_cases(
    evaluation_dir: Path,
):
    pattern = (
        STATE_PREFIX
        + "*"
        + STATE_SUFFIX
        + "*.csv"
    )

    cases = sorted(
        evaluation_dir.glob(
            pattern
        )
    )

    if len(cases) == 0:
        raise FileNotFoundError(
            f"No interpolation ignition cases found in: {evaluation_dir}"
        )

    return cases


# ============================================================
# PREDICTIONS
# ============================================================

def element_wise_prediction(
    model,
    inputs,
):
    dtype = next(
        model.parameters()
    ).dtype

    x = torch.tensor(
        inputs,
        dtype=dtype,
        device=DEVICE,
    )

    with torch.no_grad():
        prediction = model(
            x
        )

    return (
        prediction.detach()
        .cpu()
        .numpy()
    )


def autoregressive_prediction(
    model,
    first_input,
    n_steps,
):
    dtype = next(
        model.parameters()
    ).dtype

    state = torch.tensor(
        first_input,
        dtype=dtype,
        device=DEVICE,
    )

    predictions = np.zeros(
        (
            n_steps,
            len(first_input),
        ),
        dtype=float,
    )

    with torch.no_grad():

        for step in range(
            n_steps
        ):
            state = model(
                state
            )

            predictions[
                step,
                :,
            ] = (
                state.detach()
                .cpu()
                .numpy()
            )

    return predictions


# ============================================================
# IDT CALCULATION
# ============================================================

def idt_from_maximum_gradient(
    time_ms,
    values,
):
    """
    IDT = time corresponding to the maximum positive gradient.
    """

    time_ms = np.asarray(
        time_ms,
        dtype=float,
    )

    values = np.asarray(
        values,
        dtype=float,
    )

    valid = (
        np.isfinite(time_ms)
        & np.isfinite(values)
    )

    time_ms = time_ms[
        valid
    ]

    values = values[
        valid
    ]

    if len(time_ms) < 2:
        return np.nan

    gradient = np.gradient(
        values,
        time_ms,
    )

    index = int(
        np.nanargmax(
            gradient
        )
    )

    return float(
        time_ms[index]
    )


# ============================================================
# ONE CASE
# ============================================================

def evaluate_case(
    csv_path,
    model,
    means,
    stds,
):
    cantera = pd.read_csv(
        csv_path
    )

    log_dataset, columns, time = (
        hydrogen_log_transform(
            cantera
        )
    )

    if "T[K]" not in columns:
        raise ValueError(
            f"T[K] not found in {csv_path}"
        )

    if "OH" not in columns:
        raise ValueError(
            f"OH not found in {csv_path}"
        )

    dataset = z_score_and_shift(
        log_dataset,
        means,
        stds,
    )

    input_size = len(
        columns
    )

    inputs = (
        dataset.iloc[
            :,
            :input_size,
        ]
        .to_numpy()
        .astype("float32")
    )

    targets = (
        dataset.iloc[
            :,
            input_size:
            2 * input_size,
        ]
        .to_numpy()
        .astype("float32")
    )

    n_steps = (
        inputs.shape[0]
    )

    element_prediction = (
        element_wise_prediction(
            model,
            inputs,
        )
    )

    autoreg_prediction = (
        autoregressive_prediction(
            model,
            inputs[0],
            n_steps,
        )
    )

    time_ms = (
        time.iloc[1:]
        .to_numpy()
        * 1e3
    )

    temperature_index = (
        columns.index(
            "T[K]"
        )
    )

    oh_index = (
        columns.index(
            "OH"
        )
    )

    # --------------------------------------------------------
    # Reference IDT from the true temperature trajectory
    # --------------------------------------------------------

    reference_temperature = (
        inverse_transform_column(
            targets[
                :,
                temperature_index,
            ],
            temperature_index,
            means,
            stds,
        )
    )

    real_idt = (
        idt_from_maximum_gradient(
            time_ms,
            reference_temperature,
        )
    )

    # --------------------------------------------------------
    # Element-wise
    # --------------------------------------------------------

    element_temperature = (
        inverse_transform_column(
            element_prediction[
                :,
                temperature_index,
            ],
            temperature_index,
            means,
            stds,
        )
    )

    element_oh = (
        inverse_transform_column(
            element_prediction[
                :,
                oh_index,
            ],
            oh_index,
            means,
            stds,
        )
    )

    element_temperature_idt = (
        idt_from_maximum_gradient(
            time_ms,
            element_temperature,
        )
    )

    element_oh_idt = (
        idt_from_maximum_gradient(
            time_ms,
            element_oh,
        )
    )

    element_record = {
        "real_IDT_ms": real_idt,
        "temperature_IDT_ms": element_temperature_idt,
        "temperature_IDT_difference_ms": abs(
            element_temperature_idt
            - real_idt
        ),
        "OH_IDT_ms": element_oh_idt,
        "OH_IDT_difference_ms": abs(
            element_oh_idt
            - real_idt
        ),
    }

    # --------------------------------------------------------
    # Autoregressive
    # --------------------------------------------------------

    autoreg_temperature = (
        inverse_transform_column(
            autoreg_prediction[
                :,
                temperature_index,
            ],
            temperature_index,
            means,
            stds,
        )
    )

    autoreg_oh = (
        inverse_transform_column(
            autoreg_prediction[
                :,
                oh_index,
            ],
            oh_index,
            means,
            stds,
        )
    )

    autoreg_temperature_idt = (
        idt_from_maximum_gradient(
            time_ms,
            autoreg_temperature,
        )
    )

    autoreg_oh_idt = (
        idt_from_maximum_gradient(
            time_ms,
            autoreg_oh,
        )
    )

    autoreg_record = {
        "real_IDT_ms": real_idt,
        "temperature_IDT_ms": autoreg_temperature_idt,
        "temperature_IDT_difference_ms": abs(
            autoreg_temperature_idt
            - real_idt
        ),
        "OH_IDT_ms": autoreg_oh_idt,
        "OH_IDT_difference_ms": abs(
            autoreg_oh_idt
            - real_idt
        ),
    }

    return (
        element_record,
        autoreg_record,
        input_size,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    script_dir = (
        Path(__file__)
        .resolve()
        .parent
    )

    home = Path(
        environ["HOME"]
    )

    dataset_root = (
        home
        / "Alternative_trainings"
        / DATASET_FOLDER
    )

    evaluation_dir = (
        dataset_root
        / EVALUATION_CASE
    )

    means_path = (
        dataset_root
        / "mean_values_log.csv"
    )

    stds_path = (
        dataset_root
        / "std_values_log.csv"
    )

    if not means_path.is_file():
        raise FileNotFoundError(
            f"Missing normalization file: {means_path}"
        )

    if not stds_path.is_file():
        raise FileNotFoundError(
            f"Missing normalization file: {stds_path}"
        )

    means = pd.read_csv(
        means_path
    )

    stds = pd.read_csv(
        stds_path
    )

    _, model_path = (
        detect_model(
            script_dir
        )
    )

    cases = find_cases(
        evaluation_dir
    )

    print(
        f"Interpolation cases found: {len(cases)}"
    )

    # Determine dimensions from first case.
    first_cantera = pd.read_csv(
        cases[0]
    )

    first_log_dataset, first_columns, _ = (
        hydrogen_log_transform(
            first_cantera
        )
    )

    input_size = len(
        first_columns
    )

    model = load_model(
        model_path,
        input_size,
        input_size,
    )

    element_records = []
    autoreg_records = []

    for case_index, csv_path in enumerate(
        cases
    ):

        print(
            f"Processing case {case_index}: {csv_path.name}"
        )

        element_record, autoreg_record, _ = (
            evaluate_case(
                csv_path=csv_path,
                model=model,
                means=means,
                stds=stds,
            )
        )

        element_records.append({
            "case": case_index,
            **element_record,
        })

        autoreg_records.append({
            "case": case_index,
            **autoreg_record,
        })

    output_dir = (
        script_dir
        / OUTPUT_FOLDER
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    ordered_columns = [
        "case",
        "real_IDT_ms",
        "temperature_IDT_ms",
        "temperature_IDT_difference_ms",
        "OH_IDT_ms",
        "OH_IDT_difference_ms",
    ]

    element_df = pd.DataFrame(
        element_records
    )[ordered_columns]

    autoreg_df = pd.DataFrame(
        autoreg_records
    )[ordered_columns]

    element_csv = (
        output_dir
        / "element_wise_IDT_comparison.csv"
    )

    autoreg_csv = (
        output_dir
        / "autoregression_IDT_comparison.csv"
    )

    element_df.to_csv(
        element_csv,
        index=False,
    )

    autoreg_df.to_csv(
        autoreg_csv,
        index=False,
    )

    # Also save both in a single Excel-style long CSV for convenience.
    combined_df = pd.concat(
        [
            element_df.assign(
                reconstruction="element_wise"
            ),
            autoreg_df.assign(
                reconstruction="autoregression"
            ),
        ],
        ignore_index=True,
    )

    combined_df = combined_df[
        [
            "reconstruction",
            *ordered_columns,
        ]
    ]

    combined_df.to_csv(
        output_dir
        / "all_IDT_comparisons.csv",
        index=False,
    )

    print(
        "\nIDT evaluation completed."
    )

    print(
        f"Element-wise table: {element_csv}"
    )

    print(
        f"Autoregressive table: {autoreg_csv}"
    )

    print(
        f"Combined table: {output_dir / 'all_IDT_comparisons.csv'}"
    )


if __name__ == "__main__":
    main()
