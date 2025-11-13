import numpy as np
from scipy.signal import filtfilt, butter, buttord


def highpass_filter(
    data: np.ndarray,
    sample_rate: int,
    pass_freq: float,
    stop_freq: float,
    gpass: float = 1.0,
    gstop: float = 10.0,
) -> np.ndarray:
    """
    High-pass Butterworth filter for (T, N) data.
    """
    if not (0 < stop_freq < pass_freq < sample_rate / 2):
        raise ValueError("frequencies must satisfy 0 < stop < pass < Nyquist")

    order, w_n = buttord(
        wp=pass_freq,
        ws=stop_freq,
        gpass=gpass,
        gstop=gstop,
        fs=sample_rate,
    )
    b, a = butter(order, w_n, fs=sample_rate, btype="highpass")
    return filtfilt(b, a, data, axis=0, padtype="odd")


def segment_seconds(
    data: np.ndarray,
    sample_rate: int,
    *,
    n_channels: int | None = None,
) -> np.ndarray:
    """
    Split (T, N) into 1-second epochs → (samples_per_sec, N, n_epochs).
    """
    T, N = data.shape
    if n_channels is not None and n_channels != N:
        raise ValueError(f"n_channels={n_channels} but data has {N} columns")

    seg_len = sample_rate
    n_epochs = T // seg_len
    if n_epochs == 0:
        raise ValueError("not enough samples for even a single 1s segment")

    trimmed = data[: n_epochs * seg_len, :]           # (n_epochs*seg_len, N)
    out = trimmed.reshape(n_epochs, seg_len, N)       # (E, S, N)
    return np.transpose(out, (1, 2, 0))               # (S, N, E)