import numpy as np

from typing import Literal
from macro_eeg.core.types import TimeCallable
from functools import partial
from typing import cast


def _fixed_time(time: int) -> int:
    return time


def _uniform_random_time(time: tuple[int, int]) -> int:
    return np.random.randint(time[0], time[1])


def get_time_fn(
    time: int | tuple[int, int],
    strategy: Literal["fixed", "random"] = "fixed",
) -> TimeCallable:
    if strategy == "fixed":
        if not isinstance(time, int):
            raise ValueError("for 'fixed' strategy, time must be an integer")
        return cast(TimeCallable, partial(_fixed_time, time=time))

    elif strategy == "random":
        if not (isinstance(time, tuple) and len(time) == 2 and all(isinstance(t, int) for t in time)):
            raise ValueError("for 'random' strategy, time must be a tuple of two integers")
        return cast(TimeCallable, partial(_uniform_random_time, time=time))

    else:
        raise NotImplementedError(f"time strategy '{strategy}' not implemented")
