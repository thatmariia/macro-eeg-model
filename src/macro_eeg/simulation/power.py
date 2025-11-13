import numpy as np


def power_spectrum(
    data: np.ndarray,
    sample_rate: int,
    fmin: float = 0.0,
    fmax: float = 50.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute power spectrum for EEG-like data.

    Parameters
    ----------
    data : (T, N, E)
        T = samples per epoch (e.g. 1 s),
        N = channels,
        E = epochs.
    sample_rate : int
        Hz.
    fmin : float
        min frequency to return.
    fmax : float
        max frequency to return.

    Returns
    -------
    freqs : (F,)
    power : (F, N)
    """
    T, N, E = data.shape

    df = sample_rate / T                      # frequency resolution
    nyquist = sample_rate / 2

    if fmax > nyquist:
        raise ValueError(f"fmax={fmax} exceeds Nyquist={nyquist}")

    # FFT
    # shape: (T, N, E)
    fourier = np.fft.fft(data, axis=0) / T
    freqs = np.arange(0, sample_rate, df)     # length T, but we will slice

    # one-sided power (µV^2/Hz style)
    power = np.mean(np.abs(fourier) ** 2, axis=2) * (2 / df)   # (T, N)

    # slice by freq
    mask = (freqs >= fmin) & (freqs <= fmax)
    freqs_out = freqs[mask]
    power_out = power[mask, :]

    return freqs_out, power_out