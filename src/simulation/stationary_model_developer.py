# standard imports
import sys

# external imports
import numpy as np
from scipy.linalg import eig
import matplotlib.pyplot as plt
from tqdm import tqdm

# local imports
from utils.plotting_setup import notation, PLOT_SIZE, PLOT_FORMAT


class StationaryModelDeveloper:
    """
    A class to develop a stationary vector autoregression (VAR) model from given parameters.

    Attributes
    ----------
    _nr_lags : int
        The number of lags (p) in the VAR(p) model.
    _nr_nodes : int
        The number of nodes in the model.
    _nodes : list[str]
        The list of node names.
    _distances : numpy.ndarray
        A matrix containing the distances between nodes.
    _connectivity_weights : numpy.ndarray
        The initial connectivity weights between nodes.
    _sample_rate : int
        The sample rate used for the model.
    _delay_calculator : DelayCalculator
        An instance of the :py:class:`src.simulation.delay_calculator.DelayCalculator` class used to calculate delay distributions.
    _tempx : numpy.ndarray
        The array of lag indices.
    _delays_x : numpy.ndarray
        The array of delay values based on the sample rate.
    """

    def __init__(
            self,
            nr_lags,
            nr_nodes,
            nodes,
            distances,
            connectivity_weights,
            sample_rate,
            delay_calculator,
    ):
        """
        Initializes the StationaryModelDeveloper with the provided parameters.

        Parameters
        ----------
        nr_lags : int
            The number of lags (p) in the VAR(p) model.
        nr_nodes : int
            The number of nodes in the model.
        nodes : list[str]
            The list of node names.
        distances : numpy.ndarray
            A matrix containing the distances between nodes.
        connectivity_weights : numpy.ndarray
            The initial connectivity weights between nodes.
        sample_rate : int
            The sample rate used for the model.
        delay_calculator : DelayCalculator
            An instance of the :py:class:`src.simulation.delay_calculator.DelayCalculator` class used to calculate delay distributions.
        """

        self._nr_lags = nr_lags
        self._nodes = nodes
        self._nr_nodes = nr_nodes
        self._distances = distances
        self._connectivity_weights = connectivity_weights
        self._sample_rate = sample_rate
        self._delay_calculator = delay_calculator

        self._tempx = np.arange(1, self._nr_lags + 1)

        x_sample_coeff = 1000.0 / self._sample_rate
        self._delays_x = self._tempx * x_sample_coeff

    def develop(self, verbose=False):
        """
        Develops a stationary VAR(p) model.

        It calculates the lag connectivity weights using :py:meth:`_calculate_lag_connectivity_weights`,
        and adjusts the overall connectivity weights using :py:meth:`_adjust_connectivity_weights`
        until the model becomes stationary (check with :py:meth:`_is_stationary`).

        Parameters
        ----------
        verbose : bool, optional
            If True, displays progress information during the model development (default is False).

        Returns
        -------
        numpy.ndarray
            The lag connectivity weights matrix for the stationary model.
        """

        # non_stationary = True
        stationary_iters = 0

        if verbose:
            pbar = tqdm(desc="Developing stationary", unit=" iter", ascii=True, leave=False, file=sys.stdout)

        while True:

            lag_connectivity_weights = self._calculate_lag_connectivity_weights()

            if self._is_stationary(lag_connectivity_weights):
                break

            self._adjust_connectivity_weights()
            stationary_iters += 1

            if verbose:
                pbar.update(1)
                sys.stdout.flush()

        return lag_connectivity_weights

    def _adjust_connectivity_weights(self):
        """
        Adjusts the connectivity weights by scaling them down (preserving the relative weights).
        """

        self._connectivity_weights *= 0.9

    def _is_stationary(self, lag_connectivity_weights):
        """
        Determines whether the model is stationary.

        It constructs an augmented matrix from the lag connectivity weights and checks
        if all eigenvalues are within the unit circle.

        Parameters
        ----------
        lag_connectivity_weights : numpy.ndarray
            The matrix of lag connectivity weights.

        Returns
        -------
        bool
            True if the model is stationary (i.e., all eigenvalues are within the unit circle), False otherwise.
        """

        augmented_matrix = np.vstack([
            lag_connectivity_weights,
            np.hstack([
                np.eye((self._nr_lags - 1) * self._nr_nodes),
                np.zeros(((self._nr_lags - 1) * self._nr_nodes, self._nr_nodes))
            ])
        ])

        return not any(np.abs(eig(augmented_matrix, left=False, right=False)) > 1)

    def _calculate_lag_connectivity_weights(self):
        """
        Computes the connectivity weights for each lag between all pairs of nodes
        using :py:meth:`_get_lag_distribution`.

        Returns
        -------
        numpy.ndarray
            The matrix of lag connectivity weights.
        """

        lag_connectivity_weights = np.zeros((self._nr_nodes, self._nr_lags * self._nr_nodes))

        for node1 in range(self._nr_nodes):
            for node2 in range(self._nr_nodes):
                lag_dist = self._get_lag_distribution(node1, node2)
                lag_connectivity_weights[
                    node1, np.arange(node2, self._nr_lags * self._nr_nodes, self._nr_nodes)] = lag_dist

        return lag_connectivity_weights

    def _get_lag_distribution(self, node1, node2):
        """
        Calculates the lag distribution (using :py:attr:_delay_calculator and
        :py:meth:src.simulation.delay_calculator.DelayCalculator.get_delays_distribution)
        between two nodes based on their delays and connectivity weights.
        If the nodes are the same, the distribution is set to zero.

        Parameters
        ----------
        node1 : int
            The index of the first node.
        node2 : int
            The index of the second node.

        Returns
        -------
        numpy.ndarray or int
            The lag distribution values, or 0 if the nodes are the same.
        """

        if node1 == node2:
            return 0

        delays = self._delay_calculator.get_delays_distribution(
            self._delays_x,
            self._distances[node1, node2]
        )

        return delays * self._connectivity_weights[node1, node2]

    def plot_connectivity(self, lag_connectivity_weights, plots_dir):
        """
        Visualizes the lag connectivity weights between nodes as a line plot,
        showing the relative strength of connections over different delays.

        Parameters
        ----------
        lag_connectivity_weights : numpy.ndarray
            The matrix of lag connectivity weights to be plotted.
        plots_dir : pathlib.Path
            The directory where the plots are saved.

        Raises
        ------
        AssertionError
            If the plots directory does not exist.
        """

        assert plots_dir.exists(), f"Directory not found: {plots_dir}"

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(1.5 * PLOT_SIZE, PLOT_SIZE))

        for node1 in range(self._nr_nodes):
            for node2 in range(node1 + 1, self._nr_nodes):
                y = lag_connectivity_weights[node1, np.arange(node2, self._nr_lags * self._nr_nodes, self._nr_nodes)]
                ax.plot(self._delays_x, y, label=f'{notation(self._nodes[node1])} — {notation(self._nodes[node2])}')

        #ax.set_xlabel('Axon propagation delays')
        #ax.set_ylabel('Relative number of connections')
        ax.grid(which='both')
        plt.legend(ncol=2, loc="upper right")

        # plt.show()

        path = plots_dir / f"Connectivity_weighs_across_lags.{PLOT_FORMAT}"
        fig.savefig(path)
