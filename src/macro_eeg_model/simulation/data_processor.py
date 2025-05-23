# external imports
import numpy as np
from scipy.signal import filtfilt
from scipy.signal import butter, buttord


class DataProcessor:
    """
    A class responsible for processing EEG data by filtering and segmenting it.
    """

    @staticmethod
    def filter_data(data, sample_rate, pass_frequency, stop_frequency):
        """
        Filters the data using a high-pass Butterworth filter
        based on the specified passband and stopband frequencies.

        Parameters
        ----------
        data : numpy.ndarray
            The input data to be filtered (a 2D array where rows represent time points and columns represent channels/nodes).
        sample_rate : int
            The sample rate of the data in Hz.
        pass_frequency : float
            The passband edge frequency in Hz.
        stop_frequency : float
            The stopband edge frequency in Hz.

        Returns
        -------
        numpy.ndarray
            The filtered data with the same shape as the input data.

        Raises
        ------
        AssertionError
            If the frequency values are invalid.
        """

        assert 0 < stop_frequency < pass_frequency, "Invalid frequency values."

        order, w_natural = buttord(wp=pass_frequency, ws=stop_frequency, gpass=1, gstop=10, fs=sample_rate)
        b, a = butter(order, w_natural, fs=sample_rate, btype='highpass')

        filtered_data = filtfilt(b, a, data, axis=0, padtype='odd')

        return filtered_data

    @staticmethod
    def segment_data(data, sample_rate, nr_nodes):
        """
        Segments the data into epochs of 1 second each, evenly dividing the data based on the sample rate.

        Parameters
        ----------
        data : numpy.ndarray
            The input data to be segmented (a 2D array where rows represent time points and columns represent channels/nodes).
        sample_rate : int
            The sample rate of the data in Hz.
        nr_nodes : int
            The number of nodes (channels) in the data.

        Returns
        -------
        numpy.ndarray
            A 3D array where each slice along the third dimension represents a 1-second epoch of the data.
            The shape of the array is (t_samples, nr_nodes, nr_epochs), where t_samples is the number of samples per second.
        """

        # Segment the data into epochs of 1 second each
        total_nr_samples = data.shape[0]
        t_samples = sample_rate  # Each epoch is 1 second long
        nr_epochs = total_nr_samples // t_samples
        segmented_data = np.zeros((t_samples, nr_nodes, nr_epochs))

        for t in range(nr_epochs):
            segmented_data[:, :, t] = data[t_samples * t:t_samples * (t + 1), :]

        return segmented_data
