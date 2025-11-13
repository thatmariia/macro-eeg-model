import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes


def plot_lag_connectivity(
    lag_connectivity: np.ndarray,  # (n, p*n)
    *,
    node_names: list[str],
    plot_name: str | None = None,
    ax: Axes | None = None,
):
    """
    Plot per-pair lag weights over delay.
    """
    n, pn = lag_connectivity.shape
    p = pn // n

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.figure

    for i in range(n):
        for j in range(i + 1, n):
            y = lag_connectivity[i, np.arange(j, p * n, n)]
            ax.plot(np.arange(p), y, label=f"{node_names[i]} → {node_names[j]}")

    ax.grid(True)
    ax.legend(ncol=2, loc="upper right", fontsize="small")
    ax.set_xlabel("Delay (ms)")
    ax.set_ylabel("Relative connection weight")
    ax.set_title(f"Connectivity across lags {f'({plot_name})' if plot_name else ''}")

    return fig, ax