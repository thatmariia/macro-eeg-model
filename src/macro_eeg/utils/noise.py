# standard imports
from abc import ABC, abstractmethod

# external imports
import numpy as np
import colorednoise as cn
from typing import Literal
from macro_eeg.core.types import NoiseCallable
from functools import partial
from typing import cast


import numpy as np


def _apply_covariance(noise: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """
    Apply a full covariance transform to node-wise noise.

    Parameters
    ----------
    noise : (T, N) array
        Zero-mean noise with unit variance per node (after any temporal shaping).
    cov : (N, N) array
        Target covariance matrix over nodes.

    Returns
    -------
    (T, N) array
        Noise with the desired covariance structure.

    Raises
    ------
    ValueError
        If the covariance matrix has the wrong shape or is not positive definite.
    """
    noise = np.asarray(noise, dtype=float)
    cov = np.asarray(cov, dtype=float)

    nr_nodes = noise.shape[1]
    if cov.shape != (nr_nodes, nr_nodes):
        raise ValueError(
            f"covariance matrix must have shape ({nr_nodes}, {nr_nodes}), "
            f"got {cov.shape}"
        )

    try:
        # cov = R @ R.T
        R = np.linalg.cholesky(cov).T
    except np.linalg.LinAlgError as e:
        raise ValueError("covariance matrix is not positive definite") from e

    return noise @ R


def _apply_isotropic_covariance(noise: np.ndarray, std: float) -> np.ndarray:
    """
    Apply isotropic covariance with the same std for every node.

    Parameters
    ----------
    noise : (T, N) array
        Zero-mean noise with unit variance per node.
    std : float
        Target standard deviation per node.

    Returns
    -------
    (T, N) array
        Noise scaled so each node has variance std**2 (and zero cross-covariance).
    """
    if std < 0:
        raise ValueError(f"std must be non-negative, got {std}")

    nr_nodes = noise.shape[1]
    cov = np.eye(nr_nodes) * (std**2)
    return _apply_covariance(noise, cov)


def _white_noise(nr_nodes: int, nr_samples: int, sample_rate: float, *, std: float) -> np.ndarray:
    white_noise = np.random.randn(nr_samples, nr_nodes)
    white_noise *= np.sqrt(sample_rate)
    return _apply_isotropic_covariance(white_noise, std)


def _pink_noise(nr_nodes: int, nr_samples: int, sample_rate: float, *, std: float) -> np.ndarray:
    """
    Raises
    ------
    NotImplementedError
        If the sample rate is not 1000 Hz.
    """
    if sample_rate != 1000:
        raise NotImplementedError("Pink noise only supports sample rate of 1000 Hz.")
    pink_noise = cn.powerlaw_psd_gaussian(1, (nr_nodes, nr_samples)).T
    return _apply_isotropic_covariance(pink_noise, std)


def get_noise_fn(
    std: float = 1.0,
    strategy: Literal["white", "pink"] = "white",
) -> NoiseCallable:
    if strategy == "white":
        return cast(NoiseCallable, partial(_white_noise, std=std))
    elif strategy == "pink":
        return cast(NoiseCallable, partial(_pink_noise, std=std))
    else:
        raise NotImplementedError("Noise strategy not implemented.")

