#!/usr/bin/env python3

"""
Diagnose one autoregressive ignition case using the SECOND-HIGHEST
POSITIVE LOCAL MAXIMUM of dT/dt as the ignition-delay time (IDT).

The folder/input/output logic mirrors the previous outlier diagnostic:
- automatic final-model detection: k_model_n.pth / K_model_n.pth, n=0..4
- Datasets_50/interpolation input
- selectable CASE_INDEX
- reference and autoregressive temperature reconstruction
- output under:
      IDT_second_peak_diagnostic/case_<CASE_INDEX>/

The script saves:
1. Reference vs autoregressive temperature.
2. Reference vs autoregressive dT/dt.
3. Positive local gradient peaks, with the selected second-highest peak marked.
4. Initial-temperature zoom.
5. A diagnostic CSV with time, temperatures, gradients, and peak flags.
6. A summary CSV with reference IDT and second-peak autoregressive IDT.
"""

import re
from pathlib import Path
from os import environ

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# ============================================================
# USER SETTINGS
# ============================================================

DATASET_FOLDER = "Datasets_50"
EVALUATION_CASE = "interpolation"

# Ignition case to inspect
CASE_INDEX = 0

STATE_PREFIX = "State_space_cte_pressure_T"
STATE_SUFFIX = "_phi_"

OUTPUT_FOLDER = "IDT_second_peak_diagnostic"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# PLOTTING STYLE
# ============================================================

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

            layer = nn.Linear(
                input_size,
                architecture_e[0],
            )
            nn.init.xavier_normal_(
                layer.weight
            )

            encoder.append(layer)
            encoder.append(nn.Sigmoid())
            encoder.append(
                nn.Dropout(p=dropout_p)
            )

            layer = nn.Linear(
                architecture_e[0],
                output_size,
            )
            nn.init.xavier_normal_(
                layer.weight
            )

            encoder.append(layer)
            encoder.append(nn.Sigmoid())
            encoder.append(
                nn.Dropout(p=dropout_p)
            )

        else:

            for i in range(
                len(architecture_e)
            ):

                if i == 0:
                    layer = nn.Linear(
                        input_size,
                        architecture_e[i],
                    )
                else:
                    layer = nn.Linear(
                        architecture_e[i - 1],
                        architecture_e[i],
                    )

                nn.init.xavier_normal_(
                    layer.weight
                )

                encoder.append(layer)
                encoder.append(nn.Sigmoid())
                encoder.append(
                    nn.Dropout(
                        p=dropout_p
                    )
                )

            encoder.append(
                nn.Linear(
                    architecture_e[-1],
                    output_size,
                )
            )

        self.encoder = nn.Sequential(
            *encoder
        )

    def forward(self, x):
        return self.encoder(x)


# ============================================================
# MODEL DETECTION / LOADING
# ============================================================

def detect_model(script_dir: Path):

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
            "k_model_n.pth or K_model_n.pth with n in [0, 4]."
        )

    if len(candidates) > 1:
        raise RuntimeError(
            "More than one final model was found: "
            + ", ".join(
                str(path)
                for _, path in candidates
            )
        )

    model_number, model_path = (
        candidates[0]
    )

    print(
        f"Detected model: {model_path.name} "
        f"(model number={model_number})"
    )

    return (
        model_number,
        model_path,
    )


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

    model = model.to(
        DEVICE
    )

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

    if isinstance(
        obj,
        nn.DataParallel,
    ):
        obj = obj.module

    if isinstance(
        obj,
        nn.Module,
    ):
        obj = obj.to(
            DEVICE
        )

        obj.eval()

        return obj

    if isinstance(
        obj,
        dict,
    ):
        return rebuild_model_from_state_dict(
            obj,
            input_size,
            output_size,
        )

    raise TypeError(
        f"Unsupported model content in {model_path}"
    )


# ============================================================
# DATA HANDLING
# ============================================================

def hydrogen_log_transform(
    cantera_species,
):

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

    columns = (
        data_all.columns.to_list()
    )

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

    dataset = dataset.copy()

    mean_values = means.to_numpy()
    std_values = stds.to_numpy()

    for j in range(
        dataset.shape[1]
    ):

        mean = (
            mean_values[0, j + 1]
        )

        std = (
            std_values[0, j + 1]
        )

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
            f"No interpolation cases found in: {evaluation_dir}"
        )

    return cases


# ============================================================
# AUTOREGRESSIVE ROLLOUT
# ============================================================

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
# IDT FUNCTIONS
# ============================================================

def global_max_gradient_idt(
    time_ms,
    temperature,
):
    """
    Reference IDT from the global maximum of dT/dt.
    """

    time_ms = np.asarray(
        time_ms,
        dtype=float,
    )

    temperature = np.asarray(
        temperature,
        dtype=float,
    )

    gradient = np.gradient(
        temperature,
        time_ms,
    )

    index = int(
        np.nanargmax(
            gradient
        )
    )

    return (
        float(
            time_ms[index]
        ),
        float(
            gradient[index]
        ),
        index,
        gradient,
    )


def second_positive_local_peak_idt(
    time_ms,
    temperature,
):
    """
    Autoregressive IDT from the SECOND-HIGHEST POSITIVE LOCAL MAXIMUM
    of dT/dt.

    This identifies distinct local peaks first, then ranks them by
    gradient magnitude. It does not simply select the second-largest
    gradient sample.
    """

    time_ms = np.asarray(
        time_ms,
        dtype=float,
    )

    temperature = np.asarray(
        temperature,
        dtype=float,
    )

    gradient = np.gradient(
        temperature,
        time_ms,
    )

    local_peak_mask = (
        (
            gradient[1:-1]
            > gradient[:-2]
        )
        &
        (
            gradient[1:-1]
            >= gradient[2:]
        )
        &
        (
            gradient[1:-1]
            > 0.0
        )
    )

    positive_peak_indices = (
        np.where(
            local_peak_mask
        )[0]
        + 1
    )

    if len(
        positive_peak_indices
    ) < 2:

        return (
            np.nan,
            np.nan,
            None,
            gradient,
            positive_peak_indices,
            np.array(
                [],
                dtype=int,
            ),
        )

    ranked_indices = (
        positive_peak_indices[
            np.argsort(
                gradient[
                    positive_peak_indices
                ]
            )[::-1]
        ]
    )

    selected_index = int(
        ranked_indices[1]
    )

    return (
        float(
            time_ms[
                selected_index
            ]
        ),
        float(
            gradient[
                selected_index
            ]
        ),
        selected_index,
        gradient,
        positive_peak_indices,
        ranked_indices,
    )


# ============================================================
# PLOTS
# ============================================================

def plot_temperature(
    time_ms,
    reference_temperature,
    predicted_temperature,
    reference_idt,
    second_peak_idt,
    output_path,
):

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    ax.plot(
        time_ms,
        reference_temperature,
        label="Reference",
    )

    ax.plot(
        time_ms,
        predicted_temperature,
        linestyle="--",
        label="Autoregressive",
    )

    ax.axvline(
        reference_idt,
        linestyle=":",
        label=(
            f"Reference IDT = "
            f"{reference_idt:.5f} ms"
        ),
    )

    if np.isfinite(
        second_peak_idt
    ):
        ax.axvline(
            second_peak_idt,
            linestyle="-.",
            label=(
                "Autoregressive second-peak IDT = "
                f"{second_peak_idt:.5f} ms"
            ),
        )

    ax.set_xlabel(
        "Time [ms]"
    )

    ax.set_ylabel(
        "Temperature [K]"
    )

    ax.set_title(
        "Reference vs autoregressive temperature"
    )

    ax.legend(
        frameon=False
    )

    ax.grid(
        alpha=0.2
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=600,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


def plot_gradient_peaks(
    time_ms,
    reference_gradient,
    predicted_gradient,
    reference_idt,
    positive_peak_indices,
    ranked_indices,
    selected_index,
    output_path,
):

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    ax.plot(
        time_ms,
        reference_gradient,
        label=r"Reference $dT/dt$",
    )

    ax.plot(
        time_ms,
        predicted_gradient,
        linestyle="--",
        label=r"Autoregressive $dT/dt$",
    )

    if len(
        positive_peak_indices
    ) > 0:

        ax.scatter(
            time_ms[
                positive_peak_indices
            ],
            predicted_gradient[
                positive_peak_indices
            ],
            s=28,
            label=(
                "Positive local peaks"
            ),
        )

    if len(
        ranked_indices
    ) > 0:

        first_index = int(
            ranked_indices[0]
        )

        ax.scatter(
            [
                time_ms[
                    first_index
                ]
            ],
            [
                predicted_gradient[
                    first_index
                ]
            ],
            s=85,
            marker="o",
            facecolors="none",
            edgecolors="black",
            linewidths=1.5,
            label="Highest positive peak",
        )

    if selected_index is not None:

        ax.scatter(
            [
                time_ms[
                    selected_index
                ]
            ],
            [
                predicted_gradient[
                    selected_index
                ]
            ],
            s=90,
            marker="x",
            linewidths=2.0,
            label=(
                "Selected second-highest peak"
            ),
        )

        ax.axvline(
            time_ms[
                selected_index
            ],
            linestyle="-.",
            linewidth=1.4,
        )

    ax.axvline(
        reference_idt,
        linestyle=":",
        linewidth=1.4,
        label=(
            f"Reference IDT = "
            f"{reference_idt:.5f} ms"
        ),
    )

    ax.set_xlabel(
        "Time [ms]"
    )

    ax.set_ylabel(
        r"$dT/dt$ [K/ms]"
    )

    ax.set_title(
        "Temperature-gradient peak diagnostic"
    )

    ax.legend(
        frameon=False
    )

    ax.grid(
        alpha=0.2
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=600,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


def plot_initial_zoom(
    time_ms,
    reference_temperature,
    predicted_temperature,
    output_path,
):

    n_zoom = min(
        30,
        len(
            time_ms
        ),
    )

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    ax.plot(
        time_ms[
            :n_zoom
        ],
        reference_temperature[
            :n_zoom
        ],
        marker="o",
        label="Reference",
    )

    ax.plot(
        time_ms[
            :n_zoom
        ],
        predicted_temperature[
            :n_zoom
        ],
        marker="o",
        linestyle="--",
        label="Autoregressive",
    )

    ax.set_xlabel(
        "Time [ms]"
    )

    ax.set_ylabel(
        "Temperature [K]"
    )

    ax.set_title(
        "Initial autoregressive temperature behavior"
    )

    ax.legend(
        frameon=False
    )

    ax.grid(
        alpha=0.2
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=600,
        bbox_inches="tight",
    )

    plt.close(
        fig
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

    means = pd.read_csv(
        means_path
    )

    stds = pd.read_csv(
        stds_path
    )

    _, model_path = detect_model(
        script_dir
    )

    cases = find_cases(
        evaluation_dir
    )

    if (
        CASE_INDEX < 0
        or CASE_INDEX >= len(
            cases
        )
    ):

        raise IndexError(
            f"CASE_INDEX={CASE_INDEX} is invalid. "
            f"Available cases: 0 to {len(cases)-1}"
        )

    csv_path = cases[
        CASE_INDEX
    ]

    print(
        f"\nAnalyzing case {CASE_INDEX}: "
        f"{csv_path.name}"
    )

    cantera = pd.read_csv(
        csv_path
    )

    (
        log_dataset,
        columns,
        time,
    ) = hydrogen_log_transform(
        cantera
    )

    if "T[K]" not in columns:
        raise ValueError(
            "T[K] not found in dataset columns."
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
        .astype(
            "float32"
        )
    )

    targets = (
        dataset.iloc[
            :,
            input_size:
            2 * input_size,
        ]
        .to_numpy()
        .astype(
            "float32"
        )
    )

    model = load_model(
        model_path,
        input_size,
        input_size,
    )

    predictions = (
        autoregressive_prediction(
            model,
            inputs[0],
            inputs.shape[0],
        )
    )

    temperature_index = (
        columns.index(
            "T[K]"
        )
    )

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

    predicted_temperature = (
        inverse_transform_column(
            predictions[
                :,
                temperature_index,
            ],
            temperature_index,
            means,
            stds,
        )
    )

    time_ms = (
        time.iloc[1:]
        .to_numpy()
        * 1e3
    )

    (
        reference_idt,
        reference_max_gradient,
        reference_index,
        reference_gradient,
    ) = global_max_gradient_idt(
        time_ms,
        reference_temperature,
    )

    (
        second_peak_idt,
        second_peak_gradient,
        selected_index,
        predicted_gradient,
        positive_peak_indices,
        ranked_indices,
    ) = second_positive_local_peak_idt(
        time_ms,
        predicted_temperature,
    )

    # --------------------------------------------------------
    # Print diagnostics
    # --------------------------------------------------------

    print(
        "\nReference IDT "
        "(global maximum dT/dt):"
    )

    print(
        f"  {reference_idt:.8f} ms"
    )

    print(
        "\nAutoregressive IDT "
        "(second-highest positive local peak):"
    )

    print(
        f"  {second_peak_idt:.8f} ms"
        if np.isfinite(
            second_peak_idt
        )
        else "  Not available: fewer than two positive local peaks."
    )

    print(
        f"\nPositive local peaks detected: "
        f"{len(positive_peak_indices)}"
    )

    if len(
        ranked_indices
    ) > 0:

        print(
            "\nPositive local peaks ranked by dT/dt magnitude:"
        )

        for rank, index in enumerate(
            ranked_indices,
            start=1,
        ):

            print(
                f"{rank:2d}. "
                f"index={index:5d}, "
                f"time={time_ms[index]:.8f} ms, "
                f"T={predicted_temperature[index]:.6f} K, "
                f"dT/dt={predicted_gradient[index]:.6e} K/ms"
            )

    # --------------------------------------------------------
    # Output folder
    # --------------------------------------------------------

    output_dir = (
        script_dir
        / OUTPUT_FOLDER
        / f"case_{CASE_INDEX}"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Diagnostic CSV
    # --------------------------------------------------------

    diagnostic_df = pd.DataFrame({
        "time_ms": time_ms,
        "reference_temperature_K": reference_temperature,
        "autoregressive_temperature_K": predicted_temperature,
        "reference_dTdt_K_per_ms": reference_gradient,
        "autoregressive_dTdt_K_per_ms": predicted_gradient,
    })

    diagnostic_df[
        "is_positive_local_peak"
    ] = False

    diagnostic_df.loc[
        positive_peak_indices,
        "is_positive_local_peak",
    ] = True

    diagnostic_df[
        "is_selected_second_peak"
    ] = False

    if selected_index is not None:

        diagnostic_df.loc[
            selected_index,
            "is_selected_second_peak",
        ] = True

    diagnostic_df[
        "is_reference_IDT"
    ] = False

    diagnostic_df.loc[
        reference_index,
        "is_reference_IDT",
    ] = True

    diagnostic_df.to_csv(
        output_dir
        / "second_peak_temperature_gradient_diagnostic.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Summary CSV
    # --------------------------------------------------------

    summary = pd.DataFrame([{
        "case": CASE_INDEX,
        "case_name": csv_path.stem,
        "reference_IDT_ms": reference_idt,
        "reference_max_dTdt_K_per_ms": reference_max_gradient,
        "autoregressive_second_peak_IDT_ms": second_peak_idt,
        "autoregressive_second_peak_dTdt_K_per_ms": second_peak_gradient,
        "absolute_IDT_difference_ms": (
            abs(
                second_peak_idt
                - reference_idt
            )
            if np.isfinite(
                second_peak_idt
            )
            else np.nan
        ),
        "number_of_positive_local_peaks": len(
            positive_peak_indices
        ),
    }])

    summary.to_csv(
        output_dir
        / "second_peak_IDT_summary.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Plots
    # --------------------------------------------------------

    plot_temperature(
        time_ms,
        reference_temperature,
        predicted_temperature,
        reference_idt,
        second_peak_idt,
        output_dir
        / "temperature_reference_vs_autoregression_second_peak.png",
    )

    plot_gradient_peaks(
        time_ms,
        reference_gradient,
        predicted_gradient,
        reference_idt,
        positive_peak_indices,
        ranked_indices,
        selected_index,
        output_dir
        / "temperature_gradient_second_peak_diagnostic.png",
    )

    plot_initial_zoom(
        time_ms,
        reference_temperature,
        predicted_temperature,
        output_dir
        / "initial_temperature_zoom.png",
    )

    print(
        f"\nDiagnostics saved in: "
        f"{output_dir}"
    )


if __name__ == "__main__":
    main()
