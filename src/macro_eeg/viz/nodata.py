from matplotlib.axes import Axes
import sys

def render_no_data(
    *,
    ax: Axes,
    plot_name: str | None = None,
):
    print("No data to display.", file=sys.stderr)
    ax.text(
        0.5,
        0.5,
        "No data",
        ha="center",
        va="center",
        transform=ax.transAxes,
        fontsize=14,
        color="gray",
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)

    if plot_name:
        ax.set_title(plot_name)
