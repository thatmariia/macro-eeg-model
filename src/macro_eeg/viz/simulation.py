import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from matplotlib.axes import Axes
from .nodata import render_no_data


def plot_simulation(
    data: np.ndarray | None,  # (T, N)
    *,
    sample_rate: int,
    index_to_lbl: dict[int, str],
    stim_windows: list[tuple[int, int]] | None = None,  # [(onset_s, dur_s), ...]
    smooth: float | None = 2.0,
    plot_name: str | None = None,
    ax: Axes | None = None,
):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.figure

    if plot_name:
        title = f"Simulation: {plot_name}"
        if smooth is not None:
            title += " (smooth)"
    else:
        title = None

    if data is None or np.any(np.isnan(data)) or np.any([d is None for d in data]):
        render_no_data(ax=ax, plot_name=title)
        return fig, ax

    smooth = smooth if smooth != 0 else None

    T, N = data.shape
    t = np.arange(T) #/ sample_rate

    # mark stimuli
    if stim_windows:
        for (onset_s, dur_s) in stim_windows:
            ax.axvspan(onset_s, onset_s + dur_s, color="gray", alpha=0.2)

    for i in range(N):
        y = data[:, i]

        if smooth:
            ys = gaussian_filter1d(y, sigma=smooth)
            ax.plot(t, ys, label=index_to_lbl[i])
        else:
            ax.plot(t, y, label=index_to_lbl[i])

        ax.grid(True)
        ax.legend(loc="upper right")

    # ax.set_xlabel("Time (s)")
    ax.set_xlabel("Time")

    if title:
        ax.set_title(title)

    return fig, ax