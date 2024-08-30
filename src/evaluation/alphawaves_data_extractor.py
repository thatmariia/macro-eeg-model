# standard imports
from copy import deepcopy
import warnings

# external imports
from alphawaves.dataset import AlphaWaves
import mne
from scipy.signal import welch
from scipy.fftpack import fft, ifft
import numpy as np

# local imports
from utils.hidden_prints import HiddenPrints

warnings.filterwarnings("ignore")


class AlphaWavesDataExtractor:
    """
    A class to extract and process EEG data from the AlphaWaves dataset.

    Attributes
    ----------
    fs : int
        Sampling frequency in Hz.
    dataset : AlphaWaves
        An instance of the AlphaWaves dataset.
    subjects : list
        List of subjects in the AlphaWaves dataset.
    channels_collections : dict
        Dictionary categorizing channels into different brain regions.
    _frequencies : tuple
        Frequency range for filtering the EEG signal.
    _channels_all : list
        List of all EEG channels available in the dataset.
    _electrode_positions : dict
        Dictionary mapping EEG channels to their 3D positions.
    _subject_epochs : dict
        Cache for storing epochs data for each subject.
    """

    def __init__(self, fs, frequencies):
        """
        Initializes the AlphaWavesDataExtractor with the specified sampling frequency and frequency range.

        Parameters
        ----------
        fs : int
            Sampling frequency in Hz.
        frequencies : tuple
            Frequency range (low, high) for filtering the EEG signal.
        """

        self.fs = fs
        self._frequencies = frequencies

        self.dataset = AlphaWaves()
        self.subjects = self.dataset.subject_list

        self._channels_all = [
            'Fp1', 'Fp2', 'Fc5', 'Fz', 'Fc6', 'T7', 'Cz', 'T8', 'P7', 'P3', 'Pz', 'P4', 'P8', 'O1', 'Oz', 'O2'
        ]
        channels_fl = {"frontal lobe": [ch for ch in self._channels_all if ch.startswith('F')]}
        channels_pl = {"parietal lobe": [ch for ch in self._channels_all if ch.startswith('P')]}
        channels_ol = {"occiptal lobe": [ch for ch in self._channels_all if ch.startswith('O')]}
        channels_tl = {"temporal lobe": [ch for ch in self._channels_all if ch.startswith('T')]}
        # combine channels dicts in self.channels_collections
        self.channels_collections = {**channels_fl, **channels_pl, **channels_ol, **channels_tl}

        montage = mne.channels.make_standard_montage('standard_1005')
        positions = montage.get_positions()['ch_pos']
        positions = {key.lower(): value for key, value in positions.items()}

        self._electrode_positions = {
            ch: positions[ch.lower()] for ch in self._channels_all
        }

        self._subject_epochs = {}

    def _get_event_data_per_subject(self, subject_id, laplacian_spline=True):
        """
        Retrieves and preprocesses the EEG data for a specific subject.

        Parameters
        ----------
        subject_id : int
            The ID of the subject to retrieve data for.
        laplacian_spline : bool, optional
            Whether to apply the Laplacian Spline spatial filter (default is True).

        Returns
        -------
        mne.Epochs
            The preprocessed epochs object for the subject.
        """

        if subject_id in self._subject_epochs:
            return self._subject_epochs[subject_id]

        subject = self.subjects[subject_id]
        raw = self.dataset._get_single_subject_data(subject)

        # filter data and resample
        raw.filter(1, self._frequencies[1], verbose=False)

        raw.resample(sfreq=self.fs, verbose=False)

        # detect the events and cut the signal into epochs
        events = mne.find_events(raw=raw, shortest_event=1, verbose=False)
        event_id = {'closed': 1, 'open': 2}
        epochs = mne.Epochs(
            raw, events, event_id, tmin=2.0, tmax=8.0, baseline=None, verbose=False
        )
        with HiddenPrints():
            epochs.load_data()
        # epochs.pick_types(eeg=True)  # deprecated (0.24.*)
        epochs.pick(picks='eeg', verbose=False)

        if laplacian_spline:

            montage = mne.channels.make_dig_montage(ch_pos=self._electrode_positions, coord_frame='head')
            epochs.set_montage(montage)
            epochs = mne.preprocessing.compute_current_source_density(epochs, verbose=False)

        self._subject_epochs[subject_id] = deepcopy(epochs)

        return epochs

    def _get_time_series_per_subject_channel_event(self, subject_event_data=None, subject_id=None, channel='Oz', event='closed'):
        """
        Extracts time-series data for a specific channel and event.

        Parameters
        ----------
        subject_event_data : mne.Epochs, optional
            The preprocessed epochs data for a subject (default is None).
            If not provided, the data will be retrieved using :py:meth:`_get_event_data_per_subject`.
        subject_id : int, optional
            The ID of the subject to retrieve data for (default is None).
        channel : str, optional
            The EEG channel to extract data from (default is 'Oz').
        event : str, optional
            The event type ('closed' or 'open') (default is 'closed').

        Returns
        -------
        np.ndarray
            Time-series data for the specified channel and event.

        Raises
        ------
        AssertionError
            If both `subject_event_data` and `subject_id` are not provided.
            One of them must be provided to retrieve the data.
        """

        if subject_event_data is None:
            assert subject_id is not None, "Either subject_event_data or subject_id must be provided"
            subject_event_data = self._get_event_data_per_subject(subject_id)

        subject_event_data_copy = deepcopy(subject_event_data)
        # subject_event_data_copy.pick_channels([channel])  # deprecated (0.24.*)
        subject_event_data_copy.pick(picks=[channel], verbose=False)
        x_event = subject_event_data_copy[event].get_data()

        return x_event

    def _get_psd_per_subject_channel_event(self, subject_event_data=None, subject_id=None, channel='Oz', event='closed'):
        """
        Computes the Power Spectral Density (PSD) for a specific channel and event.

        Parameters
        ----------
        subject_event_data : mne.Epochs, optional
            The preprocessed epochs data for a subject (default is None).
        subject_id : int, optional
            The ID of the subject to retrieve data for (default is None).
        channel : str, optional
            The EEG channel to compute PSD for (default is 'Oz').
        event : str, optional
            The event type ('closed' or 'open') (default is 'closed').

        Returns
        -------
        tuple
            Frequencies and PSD values for the specified channel and event.
        """

        x_event = deepcopy(self._get_time_series_per_subject_channel_event(
            subject_event_data=subject_event_data, subject_id=subject_id, channel=channel, event=event
        ))
        f, s_event = welch(x_event, fs=self.fs, axis=-1)
        # s_event = np.mean(S_event, axis=0).squeeze()

        return f, s_event

    def get_time_series_per_subject_collection_event(self, subject_id, channels_collection, event='closed'):
        """
        Extracts the data for a specific channel using
        :py:meth:`_get_time_series_per_subject_channel_event` and computes
        the PSD using :py:meth:`_get_psd_per_subject_channel_event`.
        It then averages the time-series and PSD over a collection of channels for a
        specific subject and event.

        Parameters
        ----------
        subject_id : int
            The ID of the subject to retrieve data for.
        channels_collection : list
            List of EEG channels to include in the analysis.
        event : str, optional
            The event type ('closed' or 'open') (default is 'closed').

        Returns
        -------
        tuple
            Averaged time-series, frequency values, and averaged PSD for the specified collection of channels and event.
        """

        subject_event_data = self._get_event_data_per_subject(subject_id)
        ffts = []
        psds = []
        f = None

        for channel in channels_collection:
            x_event = self._get_time_series_per_subject_channel_event(
                subject_event_data=subject_event_data, subject_id=subject_id, channel=channel, event=event).squeeze()
            # flatten X_event time series
            x_event = x_event.flatten()
            epoch_len_sec = 2
            nr_epochs = int(len(x_event) / self.fs) // epoch_len_sec
            # reshape X_event time series to (k, len(X_event) // k)
            x_event = x_event[:nr_epochs * (len(x_event) // nr_epochs)].reshape(nr_epochs, -1)

            fft_event = fft(x_event, axis=-1)
            ffts.append(fft_event)

            f_channel, psd = self._get_psd_per_subject_channel_event(
                subject_event_data=subject_event_data, subject_id=subject_id, channel=channel, event=event
            )

            if f is None:
                f = f_channel

            psd = np.mean(psd, axis=0).squeeze()
            psds.append(psd)

        ffts = np.array(ffts)
        psds = np.array(psds)

        # Compute the average FFT and PSD
        avg_fft = np.mean(ffts, axis=0)
        avg_psd = np.mean(psds, axis=0)

        # Inverse FFT to get the averaged time series
        avg_time_series = ifft(avg_fft, axis=-1)
        avg_time_series = np.real(avg_time_series).squeeze()

        return avg_time_series, f, avg_psd
