from __future__ import annotations
import time
import numpy as np
from macro_eeg.core import Stimulus, NodesCollection, SimulationParams
from macro_eeg.core.types import NoiseCallable
from tqdm import tqdm
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from itertools import repeat


def _run_trial_job(
    i, noise_fn, nodes, params, lag_base, lags_stim, stimuli
):
    # show_progress must be False in workers
    return _simulate_trial(noise_fn, nodes, params, lag_base, lags_stim, stimuli, False)


def _var_step(
    lag_connectivity: np.ndarray,
    history: np.ndarray,
    node_coefs: np.ndarray | None = None,
) -> np.ndarray:
    """
    Single VAR(p) step.

    Parameters
    ----------
    lag_connectivity : (N, p*N)
        Lagged connectivity matrix.
    history : (N, p)
        Columns are x_{t-1}, x_{t-2}, ..., x_{t-p}.

    Returns
    -------
    (N,)
        Next state x_t (without noise).
    """
    n, pN = lag_connectivity.shape
    p = pN // n
    # history is (N, p); we need column-major flatten
    h = history.reshape(p * n, order="F")
    res = lag_connectivity @ h

    if node_coefs is None:
        return res

    return res * node_coefs


def _combine_lags_stim(
    lags_stim: list[np.ndarray],
) -> np.ndarray:
    if len(lags_stim) == 0:
        raise ValueError("lags_stim cannot be empty")
    if len(lags_stim) == 1:
        return lags_stim[0]

    # combine by averaging
    # combined = np.mean(np.stack(lags_stim, axis=0), axis=0)
    # return combined
    raise NotImplementedError("combining multiple stimulus lags not implemented")


def _resolve_stimuli_for_trial(
    stimuli: list[Stimulus] | None,
) -> list[Stimulus] | None:
    if stimuli is None:
        return None

    resolved_stimuli = []
    for stim in stimuli:
        onset = stim.onset_ms() if callable(stim.onset_ms) else stim.onset_ms
        duration = (
            stim.duration_ms() if callable(stim.duration_ms) else stim.duration_ms
        )

        stim_copy = stim.model_copy(
            update={
                "onset_ms": onset,
                "duration_ms": duration,
            }
        )
        resolved_stimuli.append(stim_copy)

    return resolved_stimuli


def _simulate_trial(
    noise_fn: NoiseCallable,
    nodes: NodesCollection,
    params: SimulationParams,
    lag_base: np.ndarray,
    lags_stim: list[np.ndarray] | None = None,
    stimuli: list[Stimulus] | None = None,
    show_progress: bool = False,
) -> np.ndarray:

    if stimuli is None and lags_stim is not None:
        raise ValueError("lags_stim provided but stimuli is None")

    stimuli = _resolve_stimuli_for_trial(stimuli)

    if stimuli is None:
        stimuli = []

    if lags_stim is None:
        lags_stim = []

    nr_burnin = params.t_burnin * params.sample_rate
    nr_samples = int(nr_burnin + params.t_secs * params.sample_rate)
    nr_nodes = len(nodes.nodes)

    noise = noise_fn(nr_nodes, nr_samples, params.sample_rate)

    data = np.zeros((nr_samples, nr_nodes), dtype=float)

    loop_range = range(params.t_lags, nr_samples)
    if show_progress:
        loop_range = tqdm(
            loop_range,
            desc="Simulating single trial",
            unit=" sample",
            ascii=True,
            leave=False
        )

    for t in loop_range:
        # build history (N, p)
        hist = np.empty((nr_nodes, params.t_lags), dtype=float)
        for k in range(params.t_lags):
            hist[:, k] = data[t - 1 - k, :]

        active_stimuli = [s for s in stimuli if s.is_active_at(t)]
        if active_stimuli and lags_stim:
            # add stimulus VAR step
            active_lags_stim = [
                lags_stim[i]
                for i, stim in enumerate(stimuli)
                if stim.is_active_at(t)
            ]
            lag_stim_combined = _combine_lags_stim(active_lags_stim)
            x_t = _var_step(lag_stim_combined, hist)
        else:
            # add base VAR step
            x_t = _var_step(lag_base, hist)

        # add stimuli
        for stim in active_stimuli:
            if stim.stimulus_fn is None:
                continue
            stimulus = stim.stimulus_fn(params.sample_rate, t)
            target_coefs = stim.target_coefs(nodes)
            stimulus_per_node = stimulus * target_coefs
            x_t += stimulus_per_node

        # add noise
        x_t += noise[t, :]

        data[t, :] = x_t
        sys.stdout.flush()

    return data[nr_burnin :, :]


def simulate(
    noise_fn: NoiseCallable,
    nodes: NodesCollection,
    params: SimulationParams,
    lag_base: np.ndarray,
    lags_stim: list[np.ndarray] | None = None,
    stimuli: list[Stimulus] | None = None,
    nr_trials: int = 1,
    show_progress: bool = False,
    parallel_trials: int | None = None,
    cooldown: float | None = None,
) -> np.ndarray:
    if nr_trials == 1 or (parallel_trials is not None and parallel_trials <= 1):
        return _simulate_trial(
            noise_fn, nodes, params, lag_base, lags_stim, stimuli, show_progress
        )

    P = parallel_trials or min(nr_trials, max(1, (os.cpu_count() or 1)))

    with ProcessPoolExecutor(max_workers=P) as ex:
        # futs = [
        #     ex.submit(
        #         _run_trial_job,
        #         i, noise_fn, nodes, params, lag_base, lags_stim, stimuli,
        #     )
        #     for i in range(nr_trials)
        # ]
        futs = []
        for i in range(nr_trials):
            futs.append(
                ex.submit(
                    _run_trial_job,
                    i, noise_fn, nodes, params, lag_base, lags_stim, stimuli,
                )
            )
            #optional cooldown between submissions
            if cooldown:
                time.sleep(cooldown)
        datas = []
        if show_progress:
            for fut in tqdm(
                as_completed(futs),
                total=nr_trials,
                desc="Simulating trials",
                unit=" trial",
                ascii=True,
                leave=False,
            ):
                datas.append(fut.result())
        else:
            datas = [f.result() for f in futs]

    return np.mean(datas, axis=0)

