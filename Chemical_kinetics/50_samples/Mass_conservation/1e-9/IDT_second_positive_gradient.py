#!/usr/bin/env python3

"""
IDT calculation using the SECOND-HIGHEST POSITIVE LOCAL MAXIMUM of dT/dt.

This is useful when the largest positive temperature-gradient peak is an
artificial early transient and the physically meaningful ignition event is
represented by the next strongest distinct peak.

The routine identifies local maxima in dT/dt, keeps only positive peaks,
ranks them by gradient magnitude, and selects the second-highest one.

No SciPy dependency is required.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# USER SETTINGS
# ============================================================

# Optional diagnostic CSV containing at least:
#   time_ms
#   autoregressive_temperature_K
#
# Set to None if you only want to import/use the function elsewhere.
INPUT_CSV = None

TIME_COLUMN = "time_ms"
TEMPERATURE_COLUMN = "autoregressive_temperature_K"

OUTPUT_FOLDER = "second_gradient_IDT_diagnostic"


# ============================================================
# SECOND-POSITIVE-PEAK IDT
# ============================================================

def idt_from_second_positive_gradient_peak(
    time_values,
    temperature_values,
):
    """
    Calculate IDT from the second-highest positive LOCAL maximum of dT/dt.

    Parameters
    ----------
    time_values : array-like
        Time coordinate, e.g. milliseconds.
    temperature_values : array-like
        Temperature in Kelvin.

    Returns
    -------
    idt : float
        Time corresponding to the second-highest positive local maximum.
    selected_gradient : float
        dT/dt value at the selected peak.
    selected_index : int
        Index of the selected peak in the original arrays.
    gradient : np.ndarray
        Full dT/dt array.
    positive_peak_indices : np.ndarray
        Indices of all detected positive local maxima, sorted in temporal order.

    Notes
    -----
    A local maximum i satisfies:

        gradient[i] > gradient[i-1]
        gradient[i] >= gradient[i+1]

    Only peaks with gradient[i] > 0 are retained.

    "Second maximum" means the SECOND-HIGHEST peak by gradient magnitude,
    not the second peak in chronological order.
    """

    time_values = np.asarray(
        time_values,
        dtype=float,
    )

    temperature_values = np.asarray(
        temperature_values,
        dtype=float,
    )

    if time_values.ndim != 1 or temperature_values.ndim != 1:
        raise ValueError(
            "time_values and temperature_values must be one-dimensional."
        )

    if len(time_values) != len(temperature_values):
        raise ValueError(
            "time_values and temperature_values must have the same length."
        )

    if len(time_values) < 3:
        raise ValueError(
            "At least three points are required to detect local gradient peaks."
        )

    valid = (
        np.isfinite(time_values)
        & np.isfinite(temperature_values)
    )

    if not np.all(valid):
        time_values = time_values[valid]
        temperature_values = temperature_values[valid]

    gradient = np.gradient(
        temperature_values,
        time_values,
    )

    # Local maxima of dT/dt.
    local_peak_mask = (
        (gradient[1:-1] > gradient[:-2])
        & (gradient[1:-1] >= gradient[2:])
        & (gradient[1:-1] > 0.0)
    )

    positive_peak_indices = (
        np.where(local_peak_mask)[0]
        + 1
    )

    if len(positive_peak_indices) < 2:
        return (
            np.nan,
            np.nan,
            None,
            gradient,
            positive_peak_indices,
        )

    # Rank positive local peaks by gradient magnitude, descending.
    ranked_indices = positive_peak_indices[
        np.argsort(
            gradient[positive_peak_indices]
        )[::-1]
    ]

    # Largest is rank 0; second-largest is rank 1.
    selected_index = int(
        ranked_indices[1]
    )

    idt = float(
        time_values[selected_index]
    )

    selected_gradient = float(
        gradient[selected_index]
    )

    return (
        idt,
        selected_gradient,
        selected_index,
        gradient,
        positive_peak_indices,
    )


# ============================================================
# OPTIONAL DIAGNOSTIC PLOT
# ============================================================

def plot_second_peak_diagnostic(
    time_values,
    temperature_values,
    output_path,
):
    (
        idt,
        selected_gradient,
        selected_index,
        gradient,
        positive_peak_indices,
    ) = idt_from_second_positive_gradient_peak(
        time_values,
        temperature_values,
    )

    time_values = np.asarray(
        time_values,
        dtype=float,
    )

    temperature_values = np.asarray(
        temperature_values,
        dtype=float,
    )

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    ax.plot(
        time_values,
        gradient,
        label=r"$dT/dt$",
    )

    if len(positive_peak_indices) > 0:
        ax.scatter(
            time_values[positive_peak_indices],
            gradient[positive_peak_indices],
            s=28,
            label="Positive local maxima",
        )

    if selected_index is not None:
        ax.scatter(
            [time_values[selected_index]],
            [gradient[selected_index]],
            s=80,
            marker="x",
            linewidths=2,
            label=(
                "Selected second-highest peak\n"
                f"IDT = {idt:.6f}"
            ),
        )

        ax.axvline(
            idt,
            linestyle="--",
            linewidth=1.4,
        )

    ax.set_xlabel(
        "Time"
    )

    ax.set_ylabel(
        r"$dT/dt$"
    )

    ax.set_title(
        "IDT from second-highest positive temperature-gradient peak"
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

    return (
        idt,
        selected_gradient,
    )


# ============================================================
# OPTIONAL STANDALONE EXECUTION
# ============================================================

def main():

    if INPUT_CSV is None:
        print(
            "No INPUT_CSV was specified.\n"
            "Import idt_from_second_positive_gradient_peak() "
            "into your evaluation script, or set INPUT_CSV at the top."
        )
        return

    input_path = Path(
        INPUT_CSV
    )

    if not input_path.is_file():
        raise FileNotFoundError(
            f"Input CSV not found: {input_path}"
        )

    df = pd.read_csv(
        input_path
    )

    required = {
        TIME_COLUMN,
        TEMPERATURE_COLUMN,
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    time_values = pd.to_numeric(
        df[TIME_COLUMN],
        errors="coerce",
    ).to_numpy()

    temperature_values = pd.to_numeric(
        df[TEMPERATURE_COLUMN],
        errors="coerce",
    ).to_numpy()

    output_dir = Path(
        OUTPUT_FOLDER
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        idt,
        selected_gradient,
        selected_index,
        gradient,
        positive_peak_indices,
    ) = idt_from_second_positive_gradient_peak(
        time_values,
        temperature_values,
    )

    result = pd.DataFrame([{
        "second_positive_peak_IDT": idt,
        "selected_dTdt": selected_gradient,
        "selected_index": selected_index,
        "number_of_positive_local_peaks": len(
            positive_peak_indices
        ),
    }])

    result.to_csv(
        output_dir
        / "second_positive_gradient_IDT.csv",
        index=False,
    )

    plot_second_peak_diagnostic(
        time_values,
        temperature_values,
        output_dir
        / "second_positive_gradient_IDT.png",
    )

    print(
        f"Second-positive-peak IDT: {idt}"
    )

    print(
        f"Selected dT/dt: {selected_gradient}"
    )

    print(
        f"Positive local peaks detected: {len(positive_peak_indices)}"
    )


if __name__ == "__main__":
    main()
