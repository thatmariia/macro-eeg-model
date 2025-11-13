import numpy as np

from typing import Literal
from macro_eeg.core.types import StimulusCallable
from functools import partial
from typing import cast




def _sinusoid(sample_rate: float, t: np.ndarray | float, *, amplitude: float) -> np.ndarray | float:
    t_time = t / sample_rate
    return amplitude * (1 + np.sin(t_time))

def _step_pulse(sample_rate: float, t: np.ndarray | float, *, amplitude: float) -> np.ndarray | float:
    # step function (1 for 1 second, 0 for 1 second, etc)
    t_time = t / sample_rate
    return amplitude * (t_time % 2 < 1)


def _step(sample_rate: float, t: np.ndarray | float, *, amplitude: float) -> np.ndarray | float:
    ones = np.ones_like(t, dtype=float) if isinstance(t, np.ndarray) else 1.0
    return amplitude * ones


def get_stimulus_fn(
    amplitude: float,
    strategy: Literal["sinusoid", "step pulse", "step"] = "step",
) -> StimulusCallable:
    if strategy == "sinusoid":
        return cast(StimulusCallable, partial(_sinusoid, amplitude=amplitude))
    elif strategy == "step pulse":
        return cast(StimulusCallable, partial(_step_pulse, amplitude=amplitude))
    elif strategy == "step":
        return cast(StimulusCallable, partial(_step, amplitude=amplitude))
    else:
        raise NotImplementedError(f"stimulus strategy '{strategy}' not implemented")
