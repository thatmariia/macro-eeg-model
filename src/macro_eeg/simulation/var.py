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


def _run_trial_job(i, noise_fn, nodes, params, lag_base, lags_stim, stimuli):
    # show_progress must be False in workers
    return _simulate_trial(noise_fn, nodes, params, lag_base, lags_stim, stimuli, False)


def _combine_lags_stim(
    lags_stim: list[np.ndarray],
) -> np.ndarray:
    if len(lags_stim) == 0:
        raise ValueError("lags_stim cannot be empty")
    if len(lags_stim) == 1:
        return lags_stim[0]

    # combine by averaging
    combined = np.mean(np.stack(lags_stim, axis=0), axis=0)
    return combined
    # raise NotImplementedError("combining multiple stimulus lags not implemented")


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


def _build_stimulus_schedule(
    stimuli: list[Stimulus] | None,
    t_start: int,
    t_end: int,
    t_stim_origin: int,
    sample_rate,
) -> list[list[int]] | None:
    """
    Precompute, for each time step, which stimuli are active.

    Returns
    -------
    schedule : list of length t_end
        schedule[t] is a list of indices into `stimuli` that are active at t.
    """
    if stimuli is None:
        return None

    schedule: list[list[int]] = [[] for _ in range(t_end)]

    total_activations = 0
    for idx, stim in enumerate(stimuli):
        this_stim_activations = 0
        for t in range(t_start, t_end):
            ms_per_sample = 1000.0 / sample_rate
            t_ms = (t - t_stim_origin) * ms_per_sample
            if t_ms < 0:
                continue
            if stim.is_active_at(int(t_ms)):
                schedule[t].append(idx)
                total_activations += 1
                this_stim_activations += 1

        debug_err = (
            f"DEBUG: stimulus {idx} '{stim.name}' activations:",
            this_stim_activations,
            "onset_ms:",
            stim.onset_ms,
            "duration_ms:",
            stim.duration_ms,
        )
        print(*debug_err, file=sys.stderr)

    return schedule


def _precompute_target_coefs(
    stimuli: list[Stimulus] | None,
    nodes: NodesCollection,
) -> list[np.ndarray | None]:
    if stimuli is None:
        return []

    target_coefs_per_stim: list[np.ndarray | None] = []
    for stim in stimuli:
        if stim.stimulus_fn is None:
            target_coefs_per_stim.append(None)
        else:
            target_coefs_per_stim.append(stim.target_coefs(nodes))

    return target_coefs_per_stim


def _simulate_trial(
    noise_fn: NoiseCallable,
    nodes: NodesCollection,
    params: SimulationParams,
    lag_base: np.ndarray,
    lags_stim: list[np.ndarray] | None = None,
    stimuli: list[Stimulus] | None = None,
    show_progress: bool = False,
) -> tuple[np.ndarray, np.ndarray | None]:
    if stimuli is None and lags_stim is not None:
        raise ValueError("lags_stim provided but stimuli is None")

    stimuli = _resolve_stimuli_for_trial(stimuli)

    ms_per_sample = 1000.0 / params.sample_rate
    nr_burnin = int(params.burnin_ms / ms_per_sample)
    nr_samples = int((params.burnin_ms + params.sim_ms) / ms_per_sample)
    nr_nodes = len(nodes.nodes)

    noise = noise_fn(nr_nodes, nr_samples, params.sample_rate)

    sim_data = np.zeros((nr_samples, nr_nodes), dtype=float)
    stim_data = np.zeros((nr_samples, nr_nodes), dtype=float)

    # precompute constants for VAR step
    n = nr_nodes
    pN = lag_base.shape[1]
    p = pN // n
    assert p == params.lags_ms, "lag_base and params.t_lags mismatch"

    # lag offsets: [1, 2, ..., p]
    lag_offsets = np.arange(1, p + 1, dtype=int)

    t_start = params.lags_ms
    t_end = nr_samples
    t_stim_origin = t_start + nr_burnin
    stim_schedule = _build_stimulus_schedule(
        stimuli, t_start, t_end, t_stim_origin, params.sample_rate
    )
    target_coefs_per_stim = _precompute_target_coefs(stimuli, nodes)

    loop_range = range(t_start, t_end)
    print("LOOP RANGE:", t_start, t_end, file=sys.stderr)
    if show_progress:
        loop_range = tqdm(
            loop_range,
            desc="Simulating single trial",
            unit=" sample",
            ascii=True,
            leave=False,
        )

    for t in loop_range:
        hist = sim_data[t - lag_offsets, :].T

        # VAR step
        hist_flat = hist.reshape(n * p, order="F")

        # select lag matrix (base vs stimulus)
        active_indices = stim_schedule[t] if stim_schedule is not None else []
        if active_indices and lags_stim:
            active_lags_stim = [lags_stim[i] for i in active_indices]
            lag_matrix = _combine_lags_stim(active_lags_stim)
        else:
            lag_matrix = lag_base

        x_t = lag_matrix @ hist_flat

        # add stimuli
        for i in active_indices:
            stim = stimuli[i]
            if stim.stimulus_fn is None:
                continue
            t_rel_ms = (t - t_stim_origin) / ms_per_sample
            stimulus = stim.stimulus_fn(params.sample_rate, t_rel_ms)
            target_coefs = target_coefs_per_stim[i]
            if target_coefs is not None:
                stim_data[t, :] += stimulus * target_coefs
                x_t += stimulus * target_coefs

        # add noise
        x_t += noise[t, :]

        sim_data[t, :] = x_t
        if show_progress:
            sys.stdout.flush()

    if np.any(stim_data):
        return sim_data[nr_burnin:, :], stim_data[nr_burnin:, :]

    return sim_data[nr_burnin:, :], None


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
) -> tuple[np.ndarray, np.ndarray | None]:
    if nr_trials == 1 or (parallel_trials is not None and parallel_trials <= 1):
        print("Simulating single trial...", file=sys.stderr)
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
                    i,
                    noise_fn,
                    nodes,
                    params,
                    lag_base,
                    lags_stim,
                    stimuli,
                )
            )
            # optional cooldown between submissions
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
                file=sys.stdout,
            ):
                datas.append(fut.result())
        else:
            datas = [f.result() for f in futs]

    # return np.mean(datas, axis=0)
    if isinstance(datas[0], tuple):
        sim_datas, stim_datas = zip(*datas)
        return np.mean(sim_datas, axis=0), np.mean(stim_datas, axis=0)
    else:
        return np.mean(datas, axis=0), None
