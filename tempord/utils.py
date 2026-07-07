import numpy as np
import pandas as pd
from scipy.spatial import distance
import itertools
import scipy.signal


def _scale_segment(x: np.ndarray, scaling: int) -> np.ndarray:
    """
    Scale a segment of data using the specified scaling method.

    Parameters
    ----------
    x : np.ndarray
        Input array to be scaled.
    scaling : int
        Scaling method to apply:
        - 0: No scaling, return input as-is
        - 1: Min-max normalization (scales to [0, 1] range)
        - 2: Standardization (z-score normalization)

    Returns
    -------
    np.ndarray
        Scaled array with the same shape as input.
        If denominator is zero (min-max) or standard deviation is zero (standardization),
        returns an array of zeros.

    Raises
    ------
    ValueError
        If scaling is not 0, 1, or 2.

    Examples
    --------
    >>> x = np.array([1, 2, 3, 4, 5])
    >>> _scale_segment(x, scaling=0)
    array([1., 2., 3., 4., 5.])

    >>> _scale_segment(x, scaling=1)
    array([0.  , 0.25, 0.5 , 0.75, 1.  ])

    >>> _scale_segment(x, scaling=2)
    array([-1.41421356, -0.70710678,  0.        ,  0.70710678,  1.41421356])
    """
    x = np.asarray(x, dtype=float)
    if scaling == 0:
        return x
    if scaling == 1:
        mn = np.min(x)
        mx = np.max(x)
        den = mx - mn
        return (x - mn) / den if den != 0 else np.zeros_like(x)
    if scaling == 2:
        mu = np.mean(x)
        sd = np.std(x, ddof=1)
        return (x - mu) / sd if sd != 0 else np.zeros_like(x)
    raise ValueError("scaling must be 0, 1, or 2")


def _adj_r2_simple_lm(x: np.ndarray, y: np.ndarray) -> float:
    """
    Calculate the adjusted R-squared for a simple linear regression model.

    This function computes the adjusted coefficient of determination (R²) for a
    simple linear regression with one predictor variable. The adjusted R² penalizes
    the model for the number of predictors and is useful for comparing models with
    different numbers of variables.

    Parameters
    ----------
    x : np.ndarray
        Independent variable(s). A 1D array of predictor values.
    y : np.ndarray
        Dependent variable. A 1D array of response values that must have the same
        length as x.

    Returns
    -------
    float
        The adjusted R-squared value, ranging from negative infinity to 1.0.
        - Returns np.nan if n < 3 (insufficient observations).
        - Returns np.nan if the denominator (n - p - 1) is <= 0.
        - Returns 1.0 if both residual sum of squares and total sum of squares are 0.
        - Returns 0.0 if residual sum of squares is 0 but total sum of squares is not.

    Notes
    -----
    The adjusted R² is calculated as:
        adj_R² = 1 - (1 - R²) * (n - 1) / (n - p - 1)
    where:
        - R² is the coefficient of determination
        - n is the number of observations
        - p is the number of predictors (1 in this case)

    Examples
    --------
    >>> x = np.array([1, 2, 3, 4, 5])
    >>> y = np.array([2, 4, 5, 4, 5])
    >>> adj_r2 = _adj_r2_simple_lm(x, y)
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = x.size
    if n < 3:
        return np.nan

    X = np.column_stack([np.ones(n), x])
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta

    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)

    if ss_tot == 0:
        return 0.0

    r2 = 1.0 - ss_res / ss_tot
    p = 1  # one predictor
    den = n - p - 1
    if den <= 0:
        return np.nan
    return 1.0 - (1.0 - r2) * (n - 1) / den


def _dist_td(x: np.ndarray, y: np.ndarray, td_type) -> float:

    distance_metrics = {
        "euclidean": distance.euclidean,
        "manhattan": distance.cityblock,
        "chebyshev": distance.chebyshev,
        "correlation": distance.correlation,
        "cosine": distance.cosine,
    }

    if isinstance(td_type, str):
        try:
            dist = distance_metrics[td_type]
        except KeyError:
            raise ValueError(
                f'Distance "{td_type}" is not handled. Choose from: {list(distance_metrics)}'
            )
    else:  # if td_type is callable
        dist = td_type
    dist_val = dist(x, y)

    return dist_val


def signal_phase(signal, method="radians"):

    # If binary signal
    if len(set(np.array(signal)[~np.isnan(np.array(signal))])) == 2:
        phase = _signal_phase_binary(signal)
    else:
        phase = _signal_phase_prophase(signal)

    if method.lower() in ["degree", "degrees"]:
        phase = np.rad2deg(phase)
    if method.lower() in ["perc", "percent", "percents", "percentage"]:
        phase = np.rad2deg(phase) / 360
    return phase


def _signal_phase_binary(signal):
    phase = itertools.chain.from_iterable(
        np.linspace(0, 1, sum([1 for i in v]), endpoint=False)
        for _, v in itertools.groupby(signal)
    )
    phase = np.array(list(phase))

    # Convert to radiant
    phase = np.deg2rad(phase * 360)
    return phase


def _signal_phase_prophase(signal):
    pi2 = 2.0 * np.pi

    signal = np.asarray(signal, dtype=float)
    signal = signal - np.nanmean(signal)

    # Get pro-phase
    prophase = np.mod(np.angle(scipy.signal.hilbert(signal)), pi2)

    # Transform a pro-phase to a real phase
    sort_idx = np.argsort(prophase)  # Get a sorting index
    reverse_idx = np.argsort(sort_idx)  # Get index reversing sorting
    tht = pi2 * np.arange(prophase.size) / (prophase.size)  # Set up sorted real phase
    phase = tht[reverse_idx]  # Reverse the sorting of it

    return phase
