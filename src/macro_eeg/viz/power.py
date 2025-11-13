import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes


def plot_power(
    freqs: np.ndarray,
    power: np.ndarray,  # (F, N)
    *,
    index_to_lbl: dict[int, str],
    xlim: tuple[float, float] = (0, 30),
    xticks: np.ndarray | None = None,
    ax: Axes | None = None,
):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.figure

    for i in range(power.shape[1]):
        ax.plot(
            freqs,
            power[:, i],
            label=index_to_lbl[i],
        )
    ax.legend(loc="upper right")
    ax.grid(True)

    if xticks is None:
        xticks = np.arange(0, 51, 5)
    ax.set_xticks(xticks)

    ax.set_xlim(*xlim)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power (µV²/Hz)")

    return fig, ax