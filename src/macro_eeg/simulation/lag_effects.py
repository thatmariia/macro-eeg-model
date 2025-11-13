import numpy as np
from typing import Iterable, Hashable
from macro_eeg.core import (
    EdgesCollection,
    EdgeConnectivity,
    EdgeDistance,
    Stimulus,
    SimulationParams,
    NodesCollection,
)
from macro_eeg.core.types import DelayPDFCallable
from tqdm import tqdm
import sys

# key: (pdf_id, nr_lags, ms_per_sample, distance)
_DELAY_CACHE: dict[tuple[int, int, float, Hashable], np.ndarray] = {}


def _get_delay_from_cache(
    delay_pdf_fn: DelayPDFCallable,
    nr_lags: int,
    ms_per_sample: float,
    distance: float | tuple[float, float],
) -> np.ndarray:
    """
    Returns delay_pdf(delays_x, distance), cached across calls.
    """
    pdf_id = id(delay_pdf_fn)  # avoids requiring delay_pdf to be hashable

    if isinstance(distance, float):
        dist_key = round(distance, 6)
    elif isinstance(distance, tuple):
        dist_key = tuple(round(d, 6) for d in distance)

    key = (pdf_id, nr_lags, ms_per_sample, dist_key)

    if key in _DELAY_CACHE:
        return _DELAY_CACHE[key]

    lag_indices = np.arange(1, nr_lags + 1)
    delays_x = lag_indices * ms_per_sample

    delay = delay_pdf_fn(delays_x, distance)
    _DELAY_CACHE[key] = np.array(delay, copy=True)

    return _DELAY_CACHE[key]


def _check_nodes_collections(nodes_collections: Iterable[NodesCollection]) -> None:
    """Ensure all EdgesCollection share the same nodes_collection."""
    nodes_collections = list(nodes_collections)
    if not nodes_collections:
        return
    first = nodes_collections[0]
    for nc in nodes_collections[1:]:
        if nc != first:
            raise ValueError("all EdgesCollection must have the same nodes_collection")


def _check_matrix_sizes(
    matrices: Iterable[np.ndarray],
) -> None:
    """Ensure all matrices have the same shape."""
    matrices = list(matrices)
    if not matrices:
        return
    first_shape = matrices[0].shape
    for m in matrices[1:]:
        if m.shape != first_shape:
            raise ValueError("all connectivity matrices must have the same shape")


def _validate_args(
    base_connectivity: EdgesCollection,
    distances: EdgesCollection,
    stimuli: list[Stimulus] | None,
    make_base_stationary: bool,
    make_stim_stationary: bool,
) -> None:
    """Validate argument consistency and types."""
    _check_nodes_collections(
        [
            base_connectivity.nodes_collection,
            distances.nodes_collection,
        ]
        + [
            s.edge_coeffs.nodes_collection
            for s in (stimuli or [])
            if s.edge_coeffs is not None
        ]
    )

    _check_matrix_sizes(
        [
            base_connectivity.matrix,
        ]
        + [s.edge_coeffs.matrix for s in (stimuli or []) if s.edge_coeffs is not None]
    )

    if base_connectivity.edge_type is not EdgeConnectivity:
        raise ValueError("base_connectivity edges must be of type EdgeConnectivity")

    if distances.edge_type is not EdgeDistance:
        raise ValueError("distances edges must be of type EdgeDistance")

    if make_stim_stationary and stimuli is None:
        raise ValueError(
            "cannot make stimulus connectivities stationary if no stimuli are given"
        )

    if make_stim_stationary and not make_base_stationary:
        raise ValueError(
            "cannot make stimulus connectivities stationary if base is not made stationary"
        )


def _build_lag_connectivity(
    connectivity: np.ndarray,
    distances: EdgesCollection,
    delay_pdf_fn: DelayPDFCallable,
    sim_params: SimulationParams,
) -> np.ndarray:
    """
    Construct the lagged connectivity matrix from spatial distances and a delay PDF.

    Shape: (n, n * nr_lags)
    """
    n = connectivity.shape[0]
    nr_lags = sim_params.t_lags
    lag_conn = np.zeros((n, nr_lags * n), dtype=float)

    ms_per_sample = 1000.0 / sim_params.sample_rate

    nodes_col = distances.nodes_collection
    nodes = nodes_col.nodes
    node_to_index = nodes_col.node_to_index

    for i in range(n):
        node1 = nodes_col.index_to_node[i]
        for j in range(n):
            node2 = nodes_col.index_to_node[j]
            if i == j:
                continue

            distance = distances.get_edge_value_from_nodes(node1, node2, relayed=True)
            delay = _get_delay_from_cache(
                delay_pdf_fn=delay_pdf_fn,
                nr_lags=nr_lags,
                ms_per_sample=ms_per_sample,
                distance=distance,
            )

            # fill every n-th entry starting at column j
            lag_conn[i, j : nr_lags * n : n] = delay * connectivity[i, j]

    return lag_conn


def _build_augmented(lag_conn: np.ndarray, nr_lags: int) -> np.ndarray:
    """
    Build the augmented (companion) matrix of a VAR(p)-like system.

    For nr_lags == 1, this is just the lag_conn itself.
    """
    n = lag_conn.shape[0]
    if nr_lags == 1:
        return lag_conn

    lower_left = np.eye((nr_lags - 1) * n)
    lower_right = np.zeros(((nr_lags - 1) * n, n))
    return np.vstack([lag_conn, np.hstack([lower_left, lower_right])])


def _is_stationary(lag_conn: np.ndarray, nr_lags: int) -> bool:
    from scipy.linalg import eig

    aug = _build_augmented(lag_conn, nr_lags)
    vals = eig(aug, left=False, right=False)
    return not np.any(np.abs(vals) > 1.0)


def compute(
    base_connectivity: EdgesCollection,
    distances: EdgesCollection,
    delay_pdf_fn: DelayPDFCallable,
    sim_params: SimulationParams,
    stimuli: list[Stimulus] | None = None,
    make_base_stationary: bool = True,
    make_stim_stationary: bool = False,
    max_iters: int = 100,
    shrink: float = 0.9,
    show_progress: bool = False,
) -> tuple[np.ndarray, list[np.ndarray] | None]:
    _validate_args(
        base_connectivity=base_connectivity,
        distances=distances,
        stimuli=stimuli,
        make_base_stationary=make_base_stationary,
        make_stim_stationary=make_stim_stationary,
    )

    conn = base_connectivity.matrix.copy()
    stims = [
        s.edge_coeffs.matrix.copy()
        for s in (stimuli or [])
        if s.edge_coeffs is not None
    ]

    # build lag matrices once (scaling is linear, so we can shrink them directly)
    lag_base = _build_lag_connectivity(conn, distances, delay_pdf_fn, sim_params)
    lags_stim: list[np.ndarray] = [
        _build_lag_connectivity(conn * s, distances, delay_pdf_fn, sim_params)
        for s in stims
    ]

    # if we don't enforce stationarity, we're done
    if not make_base_stationary:
        return lag_base, (lags_stim if lags_stim else None)

    # iteratively shrink to enforce stationarity
    pbar = (
        tqdm(
            desc="Developing stationary",
            unit=" iter",
            ascii=True,
            leave=False,
            file=sys.stdout,
            dynamic_ncols=False,
        )
        if show_progress
        else None
    )

    for it in range(max_iters):
        if show_progress and pbar is not None:
            pbar.update(1)
            sys.stdout.flush()

        main_ok = _is_stationary(lag_base, sim_params.t_lags)

        stims_ok = True
        if make_stim_stationary and lags_stim:
            stims_ok = all(
                _is_stationary(lag_s, sim_params.t_lags) for lag_s in lags_stim
            )

        if main_ok and stims_ok:
            return lag_base, (lags_stim if lags_stim else None)

        # shrink ALL lag matrices, same factor
        lag_base *= shrink
        lags_stim = [lag_s * shrink for lag_s in lags_stim]

    raise RuntimeError(f"Could not make stationary in {max_iters} iterations")
