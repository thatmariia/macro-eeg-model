# external imports
from scipy.signal import windows
import numpy as np


class CoherenceComputer:
    """
    A class responsible for computing the coherence between signals.

    Attributes
    ----------
    fs : int
        The sampling frequency of the signals.
    _window_type : str
        The type of window used for smoothing signals before coherence computation.
    """

    def __init__(self, fs, window_type='hann'):
        """
        Initializes the CoherenceComputer with the given sampling frequency and window type.

        Parameters
        ----------
        fs : int
            The sampling frequency of the signals.
        window_type : str, optional
            The type of window to apply for smoothing the signals before coherence computation (default is 'hann').
        """

        self.fs = fs
        self._window_type = window_type

    def compute_coherence_matched(self, sig1, sig2, smooth_signals=True):
        """
        Computes the coherence between two signals using :py:meth:`_compute_coherence`,
        with an option to smooth the signals before computation using :py:meth:`_smooth_signal`.

        Parameters
        ----------
        sig1 : numpy.ndarray
            The first signal array.
        sig2 : numpy.ndarray
            The second signal array.
        smooth_signals : bool, optional
            If True, applies a smoothing window to the signals before computing coherence (default is True).

        Returns
        -------
        tuple
            A tuple containing:
            - positive_freqs (numpy.ndarray): The array of positive frequency values.
            - positive_coherence (numpy.ndarray): The coherence values corresponding to the positive frequencies.

        Raises
        ------
        AssertionError
            If the two signals do not have the same shape.
        """

        assert sig1.shape == sig2.shape, "The two signals must have the same shape."

        if sig1.ndim == 1:
            sig1 = sig1[np.newaxis, :]
            sig2 = sig2[np.newaxis, :]

        if smooth_signals:
            sig1 = np.array([self._smooth_signal(epoch) for epoch in sig1])
            sig2 = np.array([self._smooth_signal(epoch) for epoch in sig2])

        positive_freqs, positive_coherence = self._compute_coherence(sig1, sig2)
        return positive_freqs, positive_coherence

    def _compute_coherence(self, sig1, sig2):
        """
        Computes the coherence between two signals using their cross-spectrum and power spectra.

        Parameters
        ----------
        sig1 : numpy.ndarray
            The first signal array with shape (nr_epochs, n_samples).
        sig2 : numpy.ndarray
            The second signal array with the same shape as `sig1`.

        Returns
        -------
        tuple
            A tuple containing:
            - positive_freqs (numpy.ndarray): The array of positive frequency values.
            - positive_coherence (numpy.ndarray): The coherence values corresponding to the positive frequencies.
        """

        # Compute the FFT of each epoch
        fft_sig1 = np.fft.fft(sig1, axis=1)
        fft_sig2 = np.fft.fft(sig2, axis=1)

        # Compute the cross-spectrum
        cross_spectrum = np.mean(fft_sig1 * np.conj(fft_sig2), axis=0)

        # Compute the power spectra
        power_spectrum_sig1 = np.mean(np.abs(fft_sig1) ** 2, axis=0)
        power_spectrum_sig2 = np.mean(np.abs(fft_sig2) ** 2, axis=0)

        # Compute the evaluation
        coh = np.abs(cross_spectrum) ** 2 / (power_spectrum_sig1 * power_spectrum_sig2)

        # Compute the frequency values
        freqs = np.fft.fftfreq(sig1.shape[1], 1 / self.fs)

        positive_freqs = freqs[:sig1.shape[1] // 2]
        positive_coherence = coh[:sig1.shape[1] // 2]

        return positive_freqs, positive_coherence


    def _smooth_signal(self, signal):
        """
        Applies a smoothing window to a signal.

        Parameters
        ----------
        signal : numpy.ndarray
            The input signal array to be smoothed.

        Returns
        -------
        numpy.ndarray
            The smoothed signal.
        """

        window = windows.get_window(self._window_type, signal.size)
        return signal * window
