from __future__ import annotations
import time
import numpy as np
from macro_eeg.core import Stimulus, NodesCollection, SimulationParams, TimeBase
from macro_eeg.core.types import NoiseCallable
from tqdm import tqdm
# from tqdm.auto import tqdm
import sys
from concurrent.futures import as_completed, ThreadPoolExecutor, ProcessPoolExecutor
import os
from itertools import repeat
from threading import Lock
from dataclasses import dataclass
import multiprocessing as mp
import queue as pyqueue


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

        print(
            f"DEBUG: resolved stimulus '{stim.name}' onset_ms: {onset}, duration_ms: {duration}",
            file=sys.stderr,
        )

    return resolved_stimuli


def _build_stimulus_schedule(
    stimuli: list[Stimulus] | None,
    t_start: int,
    t_end: int,
    t_stim_origin: int,
    tb: TimeBase,
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

    for idx, stim in enumerate(stimuli):
        for t in range(t_start, t_end):
            t_ms = (t - t_stim_origin) * tb.ms_per_sample
            if t_ms < 0:
                continue
            if stim.is_active_at(int(t_ms)):
                schedule[t].append(idx)

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


def _var_predict(
    lag_matrix: np.ndarray,
    sim_data: np.ndarray,
    t: int,
    lag_offsets: np.ndarray,
) -> np.ndarray:
    # sim_data shape: (T, n)
    hist = sim_data[t - lag_offsets, :].T  # (n, p)
    hist_flat = hist.reshape(hist.size, order="F")  # (n*p,)
    return lag_matrix @ hist_flat  # (n,)


@dataclass(frozen=True, slots=True)
class TrialConfig:
    nr_samples: int
    t_start: int
    t_stim_origin: int
    nr_pre_cutoff: int
    lag_offsets: np.ndarray


def make_trial_config(
    params: SimulationParams, lag_base: np.ndarray, n_nodes: int
) -> TrialConfig:
    nr_lags = params.timebase.ms_to_samples(params.lags_ms)
    if lag_base.shape[1] % n_nodes != 0:
        raise ValueError("lag_base columns must be a multiple of n_nodes (n_nodes * p)")
    nr_lags_in_base = lag_base.shape[1] // n_nodes
    if nr_lags != nr_lags_in_base:
        raise ValueError(
            f"lag_base has {nr_lags_in_base} lags, but params.lags_ms corresponds to {nr_lags} lags"
        )
    if nr_lags < 1:
        raise ValueError("params.lags_ms must correspond to at least 1 sample")

    nr_burnin = params.timebase.ms_to_samples(params.burnin_ms)
    nr_sim_samples = params.timebase.ms_to_samples(params.sim_ms)
    nr_samples = nr_burnin + nr_lags + nr_sim_samples

    lag_offsets = np.arange(1, nr_lags + 1, dtype=int)
    t_start = nr_lags
    t_stim_origin = nr_lags + nr_burnin
    nr_pre_cutoff = nr_lags + nr_burnin

    return TrialConfig(
        nr_samples=nr_samples,
        t_start=t_start,
        t_stim_origin=t_stim_origin,
        nr_pre_cutoff=nr_pre_cutoff,
        lag_offsets=lag_offsets,
    )


def _simulate_trial(
    noise_fn: NoiseCallable,
    nodes: NodesCollection,
    params: SimulationParams,
    cfg: TrialConfig,
    lag_base: np.ndarray,
    lags_stim: list[np.ndarray] | None = None,
    stimuli: list[Stimulus] | None = None,
    show_progress: bool = False,
    trial_id: int = 0,
    progress_q=None,
    progress_every: int = 200,
) -> tuple[np.ndarray, np.ndarray | None]:
    # import os
    # print(f"trial {trial_id} pid={os.getpid()} ...", file=sys.stderr)

    # verify stimuli have values for onset/duration (not callables)
    if stimuli is not None:
        for stim in stimuli:
            if callable(stim.onset_ms) or callable(stim.duration_ms):
                raise ValueError(
                    "stimuli onset_ms and duration_ms must be resolved to values before calling _simulate_trial"
                )

    nr_nodes = len(nodes.nodes)

    sim_data = np.zeros((cfg.nr_samples, nr_nodes), dtype=float)
    stim_data = np.zeros((cfg.nr_samples, nr_nodes), dtype=float)

    stim_schedule = _build_stimulus_schedule(
        stimuli, cfg.t_start, cfg.nr_samples, cfg.t_stim_origin, params.timebase
    )
    target_coefs_per_stim = _precompute_target_coefs(stimuli, nodes)

    noise = noise_fn(nr_nodes, cfg.nr_samples, params.sample_rate_hz)
    sim_data[: cfg.t_start, :] = noise[: cfg.t_start, :]

    loop_range = range(cfg.t_start, cfg.nr_samples)
    pbar = None
    if show_progress:
        pbar = tqdm(
            total=loop_range.stop - loop_range.start,
            desc="Simulating single trial",
            unit=" sample",
            ascii=True,
            leave=True,
            file=sys.stdout,
            mininterval=0.1,
            miniters=1,
            dynamic_ncols=True,
        )

    for t in loop_range:
        active_indices = stim_schedule[t] if stim_schedule is not None else []
        if active_indices and lags_stim:
            active_lags_stim = [lags_stim[i] for i in active_indices]
            lag_matrix = _combine_lags_stim(active_lags_stim)
        else:
            lag_matrix = lag_base

        x_t = _var_predict(lag_matrix, sim_data, t, cfg.lag_offsets)

        # add stimuli
        for i in active_indices:
            stim = stimuli[i]
            if stim.stimulus_fn is None:
                continue
            t_rel_ms = (t - cfg.t_stim_origin) * params.timebase.ms_per_sample
            stimulus = stim.stimulus_fn(params.sample_rate_hz, t_rel_ms)
            target_coefs = target_coefs_per_stim[i]
            if target_coefs is not None:
                stim_data[t, :] += stimulus * target_coefs
                x_t += stimulus * target_coefs

        # add noise
        x_t += noise[t, :]

        sim_data[t, :] = x_t

        if pbar is not None:
            pbar.update(1)
            # Force an occasional refresh
            if (t % 50) == 0:
                pbar.refresh()
        k = t - loop_range.start + 1
        if progress_q is not None and (k % progress_every == 0):
            progress_q.put((trial_id, progress_every))

    if pbar is not None:
        pbar.close()
    if progress_q is not None:
        remaining = (loop_range.stop - loop_range.start) % progress_every
        if remaining:
            progress_q.put((trial_id, remaining))

    sim_out = sim_data[cfg.nr_pre_cutoff :, :]
    had_stim = stimuli is not None and any(st.stimulus_fn is not None for st in stimuli)
    stim_out = stim_data[cfg.nr_pre_cutoff :, :] if had_stim else None
    return sim_out, stim_out


def _run_trial_job(args):
    # unpack in worker
    return _simulate_trial(*args)


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
    if stimuli is None and lags_stim is not None:
        raise ValueError("lags_stim provided but stimuli is None")

    nr_nodes = len(nodes.nodes)
    cfg = make_trial_config(params, lag_base, nr_nodes)
    resolved_stimuli_per_trial = [
        _resolve_stimuli_for_trial(stimuli) for _ in range(nr_trials)
    ]

    if nr_trials == 1 or (parallel_trials is not None and parallel_trials <= 1):
        print("Simulating single trial...", file=sys.stderr)
        return _simulate_trial(
            noise_fn,
            nodes,
            params,
            cfg,
            lag_base,
            lags_stim,
            resolved_stimuli_per_trial[0],
            show_progress,
            0,
            None,
        )

    total_steps_per_trial = cfg.nr_samples - cfg.t_start
    total = nr_trials * total_steps_per_trial
    trial_done = [0] * nr_trials

    ctx = mp.get_context("spawn")
    manager = ctx.Manager()
    progress_q = manager.Queue()
    pbar = tqdm(total=total, desc="Simulating (all trials)", unit="step", leave=True)

    job_args = []
    for i in range(nr_trials):
        job_args.append(
            (
                noise_fn,
                nodes,
                params,
                cfg,
                lag_base,
                lags_stim,
                resolved_stimuli_per_trial[i],
                False,  # show_progress in worker (keep False)
                i,  # trial_id
                progress_q,  # queue
                200,  # progress_every
            )
        )

    P = parallel_trials or min(nr_trials, max(1, (os.cpu_count() or 1)))

    with ProcessPoolExecutor(max_workers=P, mp_context=ctx) as ex:

        futs = [ex.submit(_run_trial_job, a) for a in job_args]
        # if cooldown:
        #     time.sleep(cooldown)
        pending = set(futs)
        tick = 0

        while pending:
            # Drain progress messages
            drained_any = False
            while True:
                try:
                    tid, n = progress_q.get_nowait()
                except pyqueue.Empty:
                    break
                drained_any = True
                trial_done[tid] += n
                pbar.update(n)
                tick += 1
                if tick % 25 == 0:
                    # cheap-ish status
                    pbar.set_postfix({
                        "last": f"T{tid}",
                        "done_trials": sum(d >= total_steps_per_trial for d in trial_done), 
                    })

            # Collect finished futures (non-blocking)
            finished = {f for f in pending if f.done()}
            for f in finished:
                # will re-raise worker exceptions here
                f.result()
            pending -= finished

            # Give the notebook UI time to repaint
            if not drained_any:
                time.sleep(0.01)

        datas = [f.result() for f in as_completed(futs)]
    pbar.close()

    sim_datas, stim_datas = zip(*datas)
    sim_mean = np.mean(sim_datas, axis=0)
    if np.any([s is None for s in stim_datas]):
        stim_mean = None
    else:
        stim_mean = np.mean(stim_datas, axis=0)

    manager.shutdown()
    return sim_mean, stim_mean
