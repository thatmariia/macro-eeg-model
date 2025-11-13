import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from matplotlib.axes import Axes


def plot_simulation(
    data: np.ndarray,  # (T, N)
    *,
    sample_rate: int,
    index_to_lbl: dict[int, str],
    stim_windows: list[tuple[int, int]] | None = None,  # [(onset_s, dur_s), ...]
    smooth: float | None = 2.0,
    ax: Axes | None = None,
):
    T, N = data.shape
    t = np.arange(T) / sample_rate

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.figure

    # mark stimuli
    if stim_windows:
        for (onset_s, dur_s) in stim_windows:
            ax.axvspan(onset_s, onset_s + dur_s, color="gray", alpha=0.2)

    for i in range(N):
        y = data[:, i]

        if smooth:
            ys = gaussian_filter1d(y, sigma=smooth)
            ax.plot(t, ys, label=f"{index_to_lbl[i]} (smooth)")
        else:
            ax.plot(t, y, label=index_to_lbl[i], alpha=0.25)

        ax.grid(True)
        ax.legend(loc="upper right")

    ax.set_xlabel("Time (s)")

    return fig, ax