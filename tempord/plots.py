import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def make_plot(
    tempord_values: pd.DataFrame,
    max_df: pd.DataFrame,
    col1: str,
    col2: str,
    max_shift_seconds: tuple,
    fs: int,
    signal_length_samples: int,
    method: str,
    td_type: str = "",
) -> plt.Figure:
    """
    Creates a heatmap visualization of temporal order values with maximum shift curve overlay.

    Parameters
    ----------
    tempord_values : pd.DataFrame
        DataFrame containing temporal order values where columns represent time points
        and rows represent shift values.
    max_df : pd.DataFrame
        DataFrame with columns 'Points' (time points) and 'Best' (shift at extremum).
    col1 : str
        Name of the first signal used in the temporal order analysis.
    col2 : str
        Name of the second signal used in the temporal order analysis.
    max_shift_seconds : tuple
        Tuple of (min_shift, max_shift) in seconds defining the range of shifts to display.
    fs : int
        Sampling frequency in Hz used for converting between samples and time units.
    signal_length_samples : int
        Total length of the signal in samples.
    method : str
        "LM" or "TD". Controls the colorbar upper bound (1 for LM, data max for TD).
    td_type : str, optional
        Distance metric name used with TD method, shown in the plot title.

    Returns
    -------
    plt.Figure
        Matplotlib figure object containing the temporal order heatmap with overlaid
        maximum values curve.

    Notes
    -----
    The heatmap uses a diverging colormap (RdBu) with vmin=0.
    Y-axis shows shift values in seconds. X-axis unit adapts to signal duration:
    seconds (<0.5 min), every-15 s (0.5–1 min), minutes (1–30 min),
    every-15 min (30–60 min), or hours (>60 min).
    A black line plots the shift at the extremum across all time points.
    """

    fig, ax = plt.subplots()

    # Extract numeric axis values from DataFrame
    x_vals = tempord_values.columns.to_numpy(dtype=float)
    y_vals = tempord_values.index.to_numpy(dtype=float)

    if method == "LM":
        v_max = 1
    else:
        v_max = tempord_values.max().max()
    # Display heatmap
    im = ax.imshow(
        tempord_values.values,
        aspect="auto",
        extent=[x_vals.min(), x_vals.max(), y_vals.min(), y_vals.max()],
        origin="lower",  # First row is drawn at the bottom
        cmap="RdBu",
        vmin=0,
        vmax=v_max,
    )
    fig.colorbar(im, ax=ax)

    # Plot the curve of maximum values across shifts
    ax.plot(
        max_df["Points"].values,
        max_df["Best"].values,
        linewidth=1.5,
        color="k",
    )
    # Axis labels and title
    ax.set_ylabel("Shift [s]")

    ax.set_title(
        f"Temporal orders ({method}{' - ' + td_type if td_type else ''}) - Signal 1: {col1}, Signal 2: {col2}"
    )

    # Y ticks: seconds
    labels_y = np.arange(max_shift_seconds[0], max_shift_seconds[1] + 1, 1, dtype=int)
    breaks_y = labels_y * fs
    ax.set_yticks(breaks_y)
    ax.set_yticklabels([str(x) for x in labels_y])

    # X ticks: adaptive resolution based on signal duration, clipped to plotted extent
    x_min, x_max = int(x_vals.min()), int(x_vals.max())
    minutes = signal_length_samples / (fs * 60)

    if minutes < 0.5:
        step = fs                  # every second
        unit_samples = fs
        xlabel = "Time [s]"
    elif minutes < 1:
        step = fs * 15             # every 15 seconds
        unit_samples = fs
        xlabel = "Time [s]"
    elif minutes <= 30:
        step = fs * 60             # every minute
        unit_samples = fs * 60
        xlabel = "Time [min]"
    elif minutes <= 60:
        step = fs * 60 * 15        # every 15 minutes
        unit_samples = fs * 60
        xlabel = "Time [min]"
    else:
        step = fs * 3600           # every hour
        unit_samples = fs * 3600
        xlabel = "Time [h]"

    breaks_x = [b for b in range(0, signal_length_samples + 1, step)]
    print(breaks_x)
    print(x_min, x_max)
    print(signal_length_samples)
    # if not breaks_x or breaks_x[-1] != x_max:
    #     breaks_x.append(x_max)
    labels_x = [str(int(b / unit_samples)) if b % step == 0 else "" for b in breaks_x]

    ax.set_xlabel(xlabel)
    ax.set_xticks(breaks_x)
    ax.set_xticklabels(labels_x)
    ax.set_ylim(max_shift_seconds[0]*fs, max_shift_seconds[1]*fs)
    ax.set_xlim(0, signal_length_samples)

    return fig
