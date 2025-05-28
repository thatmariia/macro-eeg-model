# external imports
import numpy as np
from fooof import FOOOF


class FooofTester:
    """
    A class responsible for testing the presence of peaks in EEG power spectra using FOOOF.

    Attributes
    ----------
    frequencies : numpy.ndarray
        The array of frequencies corresponding to the power spectrum.
    fm : FOOOF
        The FOOOF object for fitting spectral models.
    """

    def __init__(self, frequencies):
        """
       Initializes the FooofTester class using FOOOF.

        Parameters
        ----------
        frequencies : numpy.ndarray
            The array of frequencies corresponding to the power spectrum.
        """

        self.frequencies = frequencies
        self.fm = FOOOF(peak_width_limits=[2, 8], max_n_peaks=10, aperiodic_mode='fixed')

    def get_peaks_positions(self, powers):
        """
        Extracts the peak positions from the power spectrum using FOOOF.

        Parameters
        ----------
        powers : numpy.ndarray
            The power spectrum from which to extract peaks.

        Returns
        -------
        numpy.ndarray
            A binary array indicating the presence of peaks at the specified frequencies.
            1 if a peak is present at a given frequency, 0 otherwise.
        """
        trimmed_powers = powers[0:len(self.frequencies)]

        self.fm.fit(np.asarray(self.frequencies[1:]), np.asarray(trimmed_powers[1:]))
        peaks = self.fm.get_params('peak_params')

        # if no peaks found, return empty list
        if np.isnan(peaks).all():
            return np.zeros(len(self.frequencies))

        peaks = np.atleast_2d(peaks)
        # convert to list of (frequency, power) tuples
        # peak_coordinates = [(cf, pw) for cf, pw, _ in peaks]

        # binary peaks (1 if the peak is present at a given frequency, 0 otherwise)
        binary_peaks = np.zeros(len(self.frequencies))
        for cf, _, _ in peaks:
            # find closest index in frequency array
            idx = np.argmin(np.abs(self.frequencies - cf))
            binary_peaks[idx] = 1

        return binary_peaks




