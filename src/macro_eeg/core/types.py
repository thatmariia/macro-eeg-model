from typing import Callable
import numpy as np

DelayPDFCallable = Callable[[np.ndarray, float | tuple[float, float]], np.ndarray]

StimulusCallable = Callable[[float, np.ndarray | float], np.ndarray | float]

NoiseCallable = Callable[[int, int, float], np.ndarray]

TimeCallable = Callable[[], int]
