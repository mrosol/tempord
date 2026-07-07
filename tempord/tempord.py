import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations
from .utils import _scale_segment, _adj_r2_simple_lm, _dist_td, signal_phase
from .plots import make_plot


def get_causal_vector(tempord_values: pd.DataFrame, method, thr: float) -> pd.DataFrame:
    """
    Extract causal vectors from temporal ordering values based on a threshold.

    For LM method, identifies the shift with the maximum value (highest R²).
    For TD method, identifies the shift with the minimum value (lowest distance).

    Parameters
    ----------
    tempord_values : pd.DataFrame
        A DataFrame containing temporal ordering values where:
        - Index represents shift values
        - Columns represent points
        - Values are numeric (float)
    method : str
        "LM" to find the shift at the maximum value, "TD" to find the shift
        at the minimum value.
    thr : float
        Threshold for filtering. For LM: columns whose maximum is below thr
        are skipped. For TD: columns whose minimum is above thr are skipped.
        thr < 0 disables thresholding entirely.

    Returns
    -------
    pd.DataFrame
        A DataFrame with two columns:
        - "Points": The column names from the input DataFrame (as float)
        - "Best": The shift at the extremum in each column,
                 or NaN if the threshold condition is not met
    """

    shifts = tempord_values.index.to_numpy(dtype=float)
    points = tempord_values.columns.to_numpy(dtype=float)

    causal_vectors = np.full(len(points), np.nan, dtype=float)

    for i, point in enumerate(points):
        col = tempord_values[point].to_numpy(dtype=float)

        if method == "LM":
            col_max = np.nanmax(col)
            if thr >= 0 and col_max < thr:
                continue
            j = int(np.nanargmax(col))
            causal_vectors[i] = shifts[j]
        else:
            col_min = np.nanmin(col)
            if thr >= 0 and col_min > thr:
                continue
            j = int(np.nanargmin(col))
            causal_vectors[i] = shifts[j]

    extremum_df = pd.DataFrame(
        {
            "Points": points.astype(float),
            "Best": causal_vectors.astype(float),
        }
    )

    return extremum_df


def tempord(
    df: pd.DataFrame,
    method: str,
    modality: str,
    thr: float,
    scaling: int,
    sig_length: float,
    max_shift_seconds: tuple,
    fs: int,
    point_time_res: int = 1,
    shift_time_res: int = 1,
    make_figure: bool = True,
    td_type: str = None,
    **kwargs,
):
    """
    Compute temporal order of signals using linear regression or time-series distance.

    Parameters
    ----------
    df : pd.DataFrame
        Input signals with each column as a separate signal.
    method : str
        "LM" for linear regression or "TD" for time-series distance.
    modality : str
        Signal modality. Use "phase" to convert signals to instantaneous phase
        before analysis. Use "raw" to use the signals as it is.
    thr : float
        Threshold for parameter values (≥0 applies thresholding; <0 means no threshold).
    scaling : int
        Scaling type: 0=none, 1=min-max, 2=z-score (segment-wise).
    sig_length : float
        Window length in seconds.
    max_shift_seconds : tuple
        (max_backward, max_forward) in seconds.
    fs : int
        Sampling frequency in Hz.
    point_time_res : int, optional
        Step of center points in samples (default: 1).
    shift_time_res : int, optional
        Step of shifts in samples (default: 1).
    make_figure : bool, optional
        Whether to create a matplotlib figure (default: True).
    td_type : str or callable, optional
        Distance metric for TD method. Accepted strings: "euclidean", "manhattan",
        "chebyshev", "correlation", "cosine". Alternatively, a callable with
        signature callable(a, b, **kwargs) -> float.
    **kwargs
        Additional keyword arguments.

    Returns
    -------
    dict
        Dictionary with signal pair tuples as keys. Each value contains:
        - "Tempord": pd.DataFrame of parameter matrix
        - "Best": pd.DataFrame with values of shifts at which maximal parameter excided threshold for each point
        - "Fig": matplotlib figure or None
    """

    pairs = list(combinations(df.columns, 2))

    results = {}

    if sig_length > len(df) / fs:
        raise ValueError("Too long signal length...")
    if max(abs(max_shift_seconds[0]), max_shift_seconds[1]) > sig_length / 2:
        raise ValueError("Shift goes beyond analyzed signal part...")
    if method not in ("LM", "TD"):
        raise ValueError("Wrong method code chosen...")
    if modality not in ("raw", "phase"):
        raise ValueError("Unknown modality. Choose 'raw' or 'phase'.")

    window_samples = int(sig_length * fs)
    shift_start = int(np.ceil(max_shift_seconds[0] * fs))
    shift_end = int(np.ceil(max_shift_seconds[1] * fs))
    shifts = np.arange(shift_start, shift_end + 1, int(shift_time_res), dtype=int)
    start_point_left = int(fs * (-max_shift_seconds[0]) + window_samples / 2)
    end_point_left = int(len(df) - fs * (max_shift_seconds[1]) - window_samples / 2)

    # Central points (in samples) of the windows
    points = np.arange(start_point_left, end_point_left + 1, point_time_res)
    for col1, col2 in pairs:

        sig1 = df[col1].to_numpy(dtype=float)
        sig2 = df[col2].to_numpy(dtype=float)

        if modality == "phase":
            sig1 = signal_phase(sig1)
            sig2 = signal_phase(sig2)
        param_mat = np.full((len(points), len(shifts)), np.nan, dtype=float)

        for idx_x, start_left in enumerate(points):
            s1_left = start_left - window_samples // 2
            s1_right = s1_left + window_samples
            for idx_y, shift in enumerate(shifts):
                s2_left = s1_left + shift
                s2_right = s1_right + shift
                if s2_left < 0 or s2_right > len(sig2):
                    continue  # skip invalid shifts
                segment1 = _scale_segment(sig1[s1_left:s1_right], scaling)
                segment2 = _scale_segment(sig2[s2_left:s2_right], scaling)
                # Compute parameter for this point and shift
                if method == "LM":
                    param_mat[idx_x, idx_y] = _adj_r2_simple_lm(segment1, segment2)
                else:
                    param_mat[idx_x, idx_y] = _dist_td(segment1, segment2, td_type)
        tempord_values = pd.DataFrame(
            param_mat.T, columns=points.astype(int), index=shifts.astype(int)
        )

        # Creating DataFrame for maxima of the parameter curve
        max_df = get_causal_vector(tempord_values, method, thr)

        fig = None
        if make_figure:
            fig = make_plot(
                tempord_values=tempord_values,
                max_df=max_df,
                col1=col1,
                col2=col2,
                max_shift_seconds=max_shift_seconds,
                fs=fs,
                signal_length_samples=len(sig1),
                method=method,
                td_type=td_type,
            )
        results[(col1, col2)] = {"Tempord": tempord_values, "Best": max_df, "Fig": fig}

    return results
