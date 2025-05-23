# standard imports
import sys

# external imports
import numpy as np
import colorednoise as cn
from tqdm import tqdm


class Simulator:
    """
    The Simulator class is responsible for simulating EEG data using a vector autoregression (VAR) model.
    It generates synthetic EEG signals based on the provided lagged connectivity weights, noise characteristics,
    and other simulation parameters.

    Attributes
    ----------
    _lag_connectivity_weights : numpy.ndarray
        The lagged connectivity weights matrix used for the VAR model.
    _sample_rate : int
        The sample rate of the simulation in Hz.
    _nr_lags : int
        The number of lags (p) in the VAR(p) model.
    _nr_nodes : int
        The number of nodes (channels) in the simulation.
    _t_secs : int
        The total time of the simulation in seconds.
    _t_burnit : int
        The burn-in time for the simulation in seconds.
    _noise_color : str
        The color of the noise to be used in the simulation ('white' or 'pink').
    _std_noise : float
        The standard deviation of the noise to be used in the simulation.
    """

    def __init__(self, lag_connectivity_weights, sample_rate, nr_lags, nr_nodes, t_secs, t_burnit, noise_color, std_noise):
        """
        Initializes the Simulator with the provided parameters.

        Parameters
        ----------
        lag_connectivity_weights : numpy.ndarray
            The lagged connectivity weights matrix used for the VAR model.
        sample_rate : int
            The sample rate of the simulation in Hz.
        nr_lags : int
            The number of lags (p) in the VAR(p) model.
        nr_nodes : int
            The number of nodes (channels) in the simulation.
        t_secs : int
            The total time of the simulation in seconds.
        t_burnit : int
            The burn-in time for the simulation in seconds.
        noise_color : str
            The color of the noise to be used in the simulation ('white' or 'pink').
        std_noise : float
            The standard deviation of the noise to be used in the simulation.
        """

        self._lag_connectivity_weights = lag_connectivity_weights
        self._sample_rate = sample_rate
        self._nr_lags = nr_lags
        self._nr_nodes = nr_nodes
        self._t_secs = t_secs
        self._t_burnit = t_burnit
        self._noise_color = noise_color
        self._std_noise = std_noise

    def simulate(self, verbose=False):
        """
        The simulation generates synthetic EEG signals by applying the VAR model to the provided
        lagged connectivity weights and adding noise.

        Parameters
        ----------
        verbose : bool, optional
            If True, displays a progress bar during the simulation (default is False).

        Returns
        -------
        numpy.ndarray
            A 2D array of shape (samples, nodes) containing the simulated EEG data.

        Raises
        ------
        ValueError
            If an invalid noise color is provided.
        AssertionError
            If any of the input parameters are invalid (e.g., non-positive values for number of lags, time, or std).
        """

        assert self._nr_lags > 0, f"Expected number of lags to be > 0, but got {self._nr_lags}"
        assert self._t_secs > 0, f"Expected simulation time to be > 0, but got {self._t_secs}"
        assert self._t_burnit >= 0, f"Expected simulation time to delete to be >= 0, but got {self._t_burnit}"
        assert self._std_noise >= 0, f"Expected noise std to be >= 0, but got {self._std_noise}"

        # approximates power of EEG
        cov = np.eye(self._nr_nodes) * (self._std_noise ** 2)

        # use Cholesky factorization to find R such that R'*R = C
        try:
            cov_cholesky = np.linalg.cholesky(cov).T
        except np.linalg.LinAlgError:
            raise ValueError('Covariance matrix is not positive definite. Try increasing std_noise.')

        # throw away the first t_burnit seconds of data to ensure model convergence
        nr_burnin = self._t_burnit * self._sample_rate

        # total number of samples required (including burn-in period)
        nr_samples = nr_burnin + self._t_secs * self._sample_rate

        # ---- NOISE
        match self._noise_color:
            case "white":
                white_noise = np.random.randn(nr_samples, self._nr_nodes)
                noise = white_noise * np.sqrt(self._sample_rate)
            case "pink":
                pink_noise = cn.powerlaw_psd_gaussian(1, (self._nr_nodes, nr_samples)).T
                noise = pink_noise
            case _:
                raise ValueError(f"Invalid noise color: {self._noise_color}. Expected 'white' or 'pink'.")

        random_noise = noise @ cov_cholesky

        # ---- SIMULATION

        simulated_eeg = np.zeros((self._nr_nodes, nr_samples))

        loop_range = range(self._nr_lags, nr_samples)
        if verbose:
            loop_range = tqdm(loop_range, desc="Simulating model", ascii=True, leave=False, file=sys.stdout)

        for t in loop_range:
            # https://numpy.org/doc/stable/user/numpy-for-matlab-users.html#
            ar_input = (
                    self._lag_connectivity_weights @
                    simulated_eeg[:, np.arange(t, t - self._nr_lags, -1)].reshape(
                        self._nr_lags * self._nr_nodes,
                        order='F'
                    )
            )

            simulated_eeg[:, t] = ar_input + random_noise[t, :]

            sys.stdout.flush()

        simulation_data = simulated_eeg[:, nr_burnin:].T

        return simulation_data
