# delay_callable.py
import numpy as np
from macro_eeg.core import DiameterDist
from macro_eeg.core.types import DelayPDFCallable
from .distributions import get_distribution


def get_delay_pdf_fn(
    diam_dist: DiameterDist,
    *,
    velocity_factor: float = 6.0,
    trunc_percent: float = 0.0,
) -> DelayPDFCallable:
    """
    Return a function (t_grid, distance) -> pdf
    that uses your inverse-GEV / inverse-GEV-sum logic.
    """

    if velocity_factor <= 0.0:
        raise ValueError("velocity_factor must be greater than 0")

    if trunc_percent < 0.0 or trunc_percent > 1.0:
        raise ValueError("trunc_percent must be between 0 and 1")

    def _scale_from_distance(dist: float) -> float:
        # old behavior: distance / velocity_factor
        return dist / velocity_factor

    def _truncate(t: np.ndarray, pdf: np.ndarray) -> np.ndarray:
        p = trunc_percent
        if p <= 0.0:
            return pdf

        total = pdf.sum()
        if total <= 0:
            return pdf

        cdf = np.cumsum(pdf) / total
        keep = 1.0 - p
        idx = np.searchsorted(cdf, keep, side="right") - 1
        if idx < 0:
            idx = 0
        truncated = pdf.copy()
        truncated[idx + 1 :] = 0.0
        return truncated

    def delay_pdf_fn(t: np.ndarray, distance: float | tuple[float, float]) -> np.ndarray:
        t = np.asarray(t, dtype=float)

        if isinstance(distance, tuple):
            d1, d2 = distance
            l1 = _scale_from_distance(float(d1))
            l2 = _scale_from_distance(float(d2))
            dist_obj = get_distribution(
                "inv_gev_sum",
                lmbd1=l1,
                lmbd2=l2,
                mu=diam_dist.location,
                sigma=diam_dist.scale,
                xi=diam_dist.shape,
            )
        else:
            l = _scale_from_distance(float(distance))
            dist_obj = get_distribution(
                "inv_gev",
                lmbd=l,
                mu=diam_dist.location,
                sigma=diam_dist.scale,
                xi=diam_dist.shape,
            )

        pdf = dist_obj.pdf(t)
        pdf = np.where(np.isnan(pdf), 0.0, pdf)
        pdf = _truncate(t, pdf)
        return pdf

    return delay_pdf_fn
