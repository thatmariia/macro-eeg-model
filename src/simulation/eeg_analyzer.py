# external imports
import numpy as np
import matplotlib.pyplot as plt

# local imports
from utils.plotting_setup import notation, PLOT_SIZE, PLOT_FORMAT


class EEGAnalyzer:
    """
    The EEGAnalyzer class is responsible computing the power spectrum of EEG data.
    """

    @staticmethod
    def calculate_power(data, sample_rate):
        """
        Applies the Fast Fourier Transform (FFT) to the EEG data to calculate the power spectrum.
        It returns the frequencies and the average power spectrum across epochs/samples per second.

        Parameters
        ----------
        data : numpy.ndarray
            The EEG data to be analyzed (a 3D array with dimensions (time, nodes, epochs)).
        sample_rate : int
            The sample rate of the EEG data in Hz.

        Returns
        -------
        tuple
            A tuple containing:
            - frequencies (numpy.ndarray): The array of frequencies corresponding to the power spectrum.
            - power (numpy.ndarray): The calculated power spectrum for each frequency and node.

        Raises
        ------
        ValueError
            If the user-defined frequencies are outside the valid range determined by the Nyquist frequency.
        """

        frequencies = [0, 50]

        # Recalculate if maximum larger than Nyquist frequency
        nyquist_frequency = (2 / 5) * sample_rate
        if frequencies[1] > nyquist_frequency:
            print(
                f"User defined maximum frequency ({frequencies[1]}) is larger than the Nyquist frequency ({nyquist_frequency})")
            print("Using Nyquist frequnecy as maximum")
            frequencies[1] = nyquist_frequency

        # Recalculate if minimum is smaller than Nyquist sampling rate
        nyquist_sampling_rate = sample_rate / data.shape[0]
        if frequencies[0] < nyquist_sampling_rate and frequencies[0] != 0:
            print(
                f"User defined minimum frequency ({frequencies[0]}) is smaller than Nyquist sampling ({nyquist_sampling_rate})")
            print("Using Nyquist sampling rate as minimum frequency")
            frequencies[0] = nyquist_sampling_rate

        if frequencies[0] <= 1:
            frequencies[0] = 0

        # Power
        fourier = np.fft.fft(data, axis=0) / data.shape[0]
        used_frequencies = np.arange(0, frequencies[1] + nyquist_sampling_rate, nyquist_sampling_rate)

        # Power in standardized units (\muV^2/Hz)
        power = np.mean(np.abs(fourier) ** 2, axis=2) * (2 / nyquist_sampling_rate)

        # Find the indices for the min and max frequency
        min_index = np.argmin(np.abs(frequencies[0] - used_frequencies))
        max_index = len(used_frequencies)

        return used_frequencies[min_index:max_index], power[min_index:max_index, :]

    @staticmethod
    def plot_power(frequencies, power, nodes, plots_dir):
        """
        Visualizes the power spectrum of the EEG data (for each node/channel) as a line plot.

        Parameters
        ----------
        frequencies : numpy.ndarray
            The array of frequencies corresponding to the power spectrum.
        power : numpy.ndarray
            The calculated power spectrum for each frequency and node.
        nodes : list[str]
            The list of node/channel names corresponding to the data.
        plots_dir : pathlib.Path
            The directory where the plots are saved.

        Raises
        ------
        AssertionError
            If the plots directory does not exist.
        """

        assert plots_dir.exists(), f"Directory not found: {plots_dir}"

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(1.5 * PLOT_SIZE, PLOT_SIZE))

        ax.plot(frequencies, power)
        #ax.set_xlabel('Frequency (Hz)')
        #ax.set_ylabel('Standardized Power (units$^2$/Hz)')
        ax.grid(which='both')
        plt.legend([notation(nodes[i]) for i in range(power.shape[1])], loc="upper right")
        plt.xticks(np.arange(0, 51, 5))
        plt.xlim([0, 30])

        # plt.show()

        path = plots_dir / f"Power_spectrum.{PLOT_FORMAT}"
        fig.savefig(path)
