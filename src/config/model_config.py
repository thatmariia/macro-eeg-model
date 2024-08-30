# standard imports
from typing import Optional

# external imports
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# local imports
from utils.plotting_setup import notation, PLOT_SIZE, PLOT_FORMAT, COLOR_MAP
from simulation.delay_calculator import DelayCalculator
from config.connectivity_model import ConnectivityModel


class ModelConfig:
    """
    A class to configure parameters and model the connectivity between brain nodes, including
    the distances, connectivity weights, and the generation of delays.

    Attributes
    ----------
    nodes : list[str]
        The list of processed nodes used in the model.
    nr_nodes : int
        The total number of nodes in the model.
    relay_station : str, optional
        The relay station node name, if any.
    sample_rate : int
        The sampling rate of the model, in Hz.
    nr_lags : int
        The number of time lags calculated based on the sample rate and the total time (ms) in lags.
    t_secs : int
        The total time of the simulation in seconds.
    t_burnit : int
        The burn-in time for the simulation, in seconds.
    noise_color : str
        The color of the noise to be used in the simulation.
    std_noise : int
        The standard deviation of the noise to be used in the simulation.
    distances : numpy.ndarray
        A matrix containing the distances between the nodes.
    connectivity_weights : numpy.ndarray
        A matrix containing the connectivity weights between the nodes.
    delay_calculator : DelayCalculator
        An instance of the :py:class:`src.simulation.delay_calculator.DelayCalculator` class used to calculate delay distributions.
    _dist_shape : float
        The shape parameter for the delay distribution (xi in GEV distribution).
    _dist_scale : float
        The scale parameter for the delay distribution (sigma in GEV distribution).
    _dist_location : float
        The location parameter for the delay distribution (mu in GEV distribution).
    _truncation_percentile : float
        The percentile at which to truncate the delay distribution.
    """

    def __init__(
            self,
            nodes: list[str],
            relay_station: Optional[str],
            sample_rate: int,
            t_lags: int,
            t_secs: int,
            t_burnit: int,
            noise_color: str,
            std_noise: int,
            dist_shape: float,
            dist_scale: float,
            dist_location: float,
            dist_trunc_percent: float,
            custom_connectivity=False
    ):
        """
        Initializes the ModelConfig with specified parameters for nodes, connectivity,
        simulation, and delay distribution.

        Parameters
        ----------
        nodes : list[str]
            The list of nodes to be used in the connectivity model.
        relay_station : str, optional
            The relay station name, if any.
        sample_rate : int
            The sampling rate of the model, in Hz.
        t_lags : int
            The total time in lags for the simulation.
        t_secs : int
            The total time of the simulation in seconds.
        t_burnit : int
            The burn-in time for the simulation, in seconds.
        noise_color : str
            The color of the noise to be used in the simulation.
        std_noise : int
            The standard deviation of the noise to be used in the simulation.
        dist_shape : float
            The shape parameter for the delay distribution (xi in GEV distribution).
        dist_scale : float
            The scale parameter for the delay distribution (sigma in GEV distribution).
        dist_location : float
            The location parameter for the delay distribution (mu in GEV distribution).
        dist_trunc_percent : float
            The percentile at which to truncate the delay distribution.
        custom_connectivity : bool, optional
            If True, use custom connectivity weights from a pre-specified file.
        """

        # given params:
        # nodes
        self.relay_station = relay_station
        self.sample_rate = sample_rate
        x_sample_coeff = 1000.0 / self.sample_rate
        self.nr_lags = int(t_lags / x_sample_coeff)
        self.t_secs = t_secs
        self.t_burnit = t_burnit
        self.noise_color = noise_color
        self.std_noise = std_noise

        # generating
        self._dist_shape, self._dist_scale, self._dist_location = dist_shape, dist_scale, dist_location
        self._truncation_percentile = dist_trunc_percent
        self.delay_calculator = DelayCalculator(
            shape_param=dist_shape,
            scale_param=dist_scale,
            location_param=dist_location,
            truncation_percentile=dist_trunc_percent
        )

        connectivity_model = ConnectivityModel(given_nodes=nodes, relay_station=self.relay_station)
        connectivity_model.set_connectivity(custom_connectivity=custom_connectivity)

        self.distances = connectivity_model.distances
        self.connectivity_weights = connectivity_model.connectivity_weights
        self.nodes = connectivity_model.nodes
        self.nr_nodes = connectivity_model.nr_nodes

    def __str__(self):
        """
        Returns a string representation of the ModelConfig object, including details
        about the nodes, connectivity, simulation parameters, and GEV distribution parameters.

        Returns
        -------
        str
            A formatted string representation of the ModelConfig object.
        """

        notation_nodes = [notation(node) for node in self.nodes]
        distances_df = pd.DataFrame(self.distances, index=notation_nodes, columns=notation_nodes)
        connectivity_weights_df = pd.DataFrame(self.connectivity_weights, index=notation_nodes, columns=notation_nodes)
        pd.set_option('display.max_columns', None)

        model_config_str = f"""
            ModelConfig
            --- NODES -----------------------------------------------
            nodes = {self.nodes}
            relay_station = {self.relay_station}
            
            --- CONNECTIVITY ----------------------------------------
            distances = 
            {distances_df}
            
            connectivity_weights = 
            {connectivity_weights_df}
            
            --- SIMULATION PARAMETERS -------------------------------
            sample_rate = {self.sample_rate}
            nr_lags     = {self.nr_lags}
            t_secs      = {self.t_secs}
            t_burnit    = {self.t_burnit}
            noise_color = {self.noise_color}
            std_noise   = {self.std_noise}
            
            --- GEV DISTRIBUTION ------------------------------------
            dist_shape (xi)     = {self._dist_shape}
            dist_scale (sigma)  = {self._dist_scale}
            dist_location (mu)  = {self._dist_location}
            dist_trunc_percent  = {self._truncation_percentile}
        """

        # return textwrap.dedent(model_config_str)
        lines = model_config_str.splitlines()
        trimmed_lines = [line.lstrip() for line in lines]
        return "\n".join(trimmed_lines)

    def plot(self, plots_dir):
        """
        Plots (using :py:meth:`_plot_properties`) the connectivity model's distances
        (summed through the relay, if applicable) and normalized weights matrices using heatmaps.

        Parameters
        ----------
        plots_dir : pathlib.Path
            The directory where the plots are saved.

        Raises
        ------
        AssertionError
            If the plots directory does not exist.
        """

        assert plots_dir.exists(), f"Directory not found: {plots_dir}"

        if not all([distance is None for distance in self.distances.flatten()]):

            if self.relay_station is None:
                distances = self.distances
                title = "Distances"
            else:
                distances = np.zeros((self.nr_nodes, self.nr_nodes))
                np.fill_diagonal(distances, np.nan)
                for i in range(self.nr_nodes):
                    for j in range(i + 1, self.nr_nodes):
                        distances[i, j] = distances[j, i] = self.distances[i, j][0] + self.distances[i, j][1]
                title = "Distances_relayed"

            distances = np.triu(distances)
            distances[distances == 0] = np.nan
            self._plot_properties(distances[:-1, 1:], title, plots_dir, factor=0.1)

        connectivity_weights = np.triu(self.connectivity_weights)
        connectivity_weights = connectivity_weights / np.nanmax(connectivity_weights)  # normalize the matrix
        connectivity_weights[connectivity_weights == 0] = np.nan
        self._plot_properties(connectivity_weights[:-1, 1:], "Connectivity_weights", plots_dir)

    def _plot_properties(self, matrix, title, plots_dir, factor=1.0):
        """
        Helper method to plot a heatmap of a given matrix with specified properties.

        Parameters
        ----------
        matrix : numpy.ndarray
            The matrix to be plotted as a heatmap.
        title : str
            The title for the plot, used to label the saved file.
        plots_dir : pathlib.Path
            The directory where the plots are saved.
        factor : float, optional
            A scaling factor applied to the matrix values (default is 1.0).
        """

        fig, ax = plt.subplots(figsize=(1.2 * PLOT_SIZE, 0.7 * PLOT_SIZE))
        notations = [notation(node) for node in self.nodes]
        ax = sns.heatmap(
            matrix * factor,
            annot=True,
            fmt="0.3f",
            annot_kws={"size": 3 * PLOT_SIZE},
            cbar=True,
            cbar_kws={
                "pad": 0.1
            },
            cmap=COLOR_MAP,
            xticklabels=notations[1:],
            yticklabels=notations[:-1],
            linewidths=PLOT_SIZE / 2,
            linecolor="#FFFFFF"
        )

        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top')
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position('right')
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        # plt.show()

        path = plots_dir / f"{title}.{PLOT_FORMAT}"
        fig.savefig(path)
