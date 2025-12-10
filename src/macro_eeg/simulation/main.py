from .var import simulate
from .process import highpass_filter, segment_seconds
from .power import power_spectrum


def simulate_and_power(
    noise_fn,
    nodes,
    params,
    lag_base,
    lags_stim,
    stimuli,
    nr_trials,
    pass_freq: float,
    stop_freq: float,
    show_progress: bool = True,
    parallel_trials=None,
    cooldown=None,
):
    sim_data, stim_data = simulate(
        noise_fn,
        nodes,
        params,
        lag_base,
        lags_stim,
        stimuli,
        nr_trials,
        show_progress,
        parallel_trials,
        cooldown,
    )

    sim_data_processed = highpass_filter(
        sim_data,
        sample_rate=params.sample_rate,
        pass_freq=pass_freq,
        stop_freq=stop_freq,
    )

    sim_data_processed = segment_seconds(
        sim_data_processed,
        sample_rate=params.sample_rate,
    )

    power_spec = power_spectrum(
        sim_data_processed, sample_rate=params.sample_rate
    )

    return sim_data, stim_data, power_spec