# standard imports
import sys

# external imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from tqdm import tqdm

# local imports
from evaluation.simulation_data_extractor import SimulationDataExtractor
from evaluation.coherence_computer import CoherenceComputer
from utils.plotting_setup import PLOT_SIZE, COLORS, notation, PLOT_FORMAT
from utils.paths import paths


class Evaluator:
    """
    A class responsible for evaluating simulated EEG data .
    It computes metrics such as coherence and power spectra across different brain regions (nodes).

    Attributes
    ----------
    frequencies : list
        The frequency range for evaluating the data ([0, 30] Hz).
    simulation_data_extractor : SimulationDataExtractor
        An instance of the :py:class:`src.evaluation.simulation_data_extractor.SimulationDataExtractor` class
        used to extract and process simulated EEG data.
    """

    def __init__(self):
        """
        Initializes the Evaluator class, setting up the frequency range and loading real and simulated data.
        """

        self.frequencies = (0, 30)
        self.simulation_data_extractor = SimulationDataExtractor()

    def evaluate(self, plot_overview=True):
        """
        Evaluates and compares the coherence and power metrics
        using :py:meth:`_evaluate_metric`.

        Parameters
        ----------
        plot_overview : bool, optional
            If True, generates overview plots for the evaluated metrics;
            if False, generates individual plots for (pairs of) brain regions.
            (default is True).
        """

        self._evaluate_metric(self._evaluate_coherence_node_pair, "Evaluating coherence", plot_overview, 3, 2,"Coherences_summary")
        self._evaluate_metric(self._evaluate_power_node, "Evaluating power", plot_overview, 2, 2, "Powers_summary")

        print(f"The evaluation plots have been saved in the 'plots' directory.")

    def _evaluate_metric(self, evaluation_func, desc, plot_overview, rows, cols, save_file_name):
        """
        A helper function to evaluate a specific metric (e.g., coherence or power) across nodes or node pairs.

        Parameters
        ----------
        evaluation_func : function
            The function to evaluate the metric
            (:py:meth:`_evaluate_coherence_node_pair` or :py:meth:`_evaluate_power_node`).
        desc : str
            The description for the tqdm progress bar.
        plot_overview : bool
            If True, generates overview plots for the evaluated metrics;
            if False, generates individual plots for (pairs of) brain regions.
        rows : int
            The number of rows in the overview plot.
        cols : int
            The number of columns in the overview plot.
        save_file_name : str
            The file name for saving the overview plot.
        """

        fig, ax = None, None
        if plot_overview:
            fig, ax = plt.subplots(
                nrows=rows, ncols=cols,
                sharex=True, sharey=True,
                figsize=(1.5 * PLOT_SIZE * cols, PLOT_SIZE * rows)
            )

        with tqdm(desc=desc, unit=" iter", ascii=True, leave=False, file=sys.stdout) as pbar:
            for plot_id, nodes in enumerate(
                    self._get_nodes(pairwise=evaluation_func == self._evaluate_coherence_node_pair)
            ):
                pbar.update(1)
                sys.stdout.flush()
                evaluation_func(
                    *nodes, fig=fig, ax=self._get_ax(ax, cols, plot_id), show_legend=(plot_id == rows * cols - 1)
                )

        if plot_overview:
            # plt.show()
            path = paths.plots_path / f"{save_file_name}.{PLOT_FORMAT}"
            fig.savefig(path)

    def _get_nodes(self, pairwise=False):
        """
        Generates nodes or node pairs for evaluation.

        Parameters
        ----------
        pairwise : bool, optional
            If True, generates pairs of nodes (for coherence evaluation),
            otherwise generates individual nodes (for power evaluation)
            (default is False).

        Yields
        ------
        tuple
            A tuple containing one or two nodes, depending on the value of `pairwise`.
        """

        nodes = list(self.simulation_data_extractor.nodes[:-1])
        if pairwise:
            for i, node1 in enumerate(nodes):
                for node2 in nodes[i + 1:]:
                    yield node1, node2
        else:
            for node in nodes:
                yield (node,)

    def _evaluate_power_node(self, node, fig=None, ax=None, show_legend=True):
        """
        Evaluates (using :py:meth:`_get_simulated_power`)
        and plots (using :py:meth:`_plot_metric`) the power spectrum for a given node.

        Parameters
        ----------
        node : str
            The name of the brain region to evaluate.
        fig : matplotlib.figure.Figure, optional
            The figure object for plotting (default is None).
        ax : matplotlib.axes.Axes, optional
            The axis object for plotting (default is None).
        show_legend : bool, optional
            If True, shows the legend on the plot (default is True).
        """

        sim_frequencies, sim_powers = self._get_simulated_power(node)

        self._plot_metric(
            f"{notation(node)}",
            sim_frequencies, sim_powers,
            fig=fig, ax=ax, show_legend=show_legend if ax is not None else True,
            y_label="Power", xlim=[self.frequencies[0], self.frequencies[1]], ylim=[0, 2e7], file_label=f"power_{node}"
        )

    def _evaluate_coherence_node_pair(self, node1, node2, fig=None, ax=None, show_legend=True):
        """
        Evaluates (using :py:meth:`_get_simulated_coherences`)
        and plots (using :py:meth:`_plot_metric`)
        the coherence between a pair of nodes.

        Parameters
        ----------
        node1 : str
            The name of the first brain region (node).
        node2 : str
            The name of the second brain region (node).
        fig : matplotlib.figure.Figure, optional
            The figure object for plotting (default is None).
        ax : matplotlib.axes.Axes, optional
            The axis object for plotting (default is None).
        show_legend : bool, optional
            If True, shows the legend on the plot (default is True).
        """

        # simulated data
        sim_frequencies_coherence, sim_coherences = self._get_simulated_coherences(node1, node2)

        self._plot_metric(
            f"{notation(node1)} — {notation(node2)}",
            sim_frequencies_coherence, sim_coherences,
            fig=fig, ax=ax, show_legend=show_legend if ax is not None else True,
            y_label="Coherence", xlim=[1, self.frequencies[1]], ylim=[0, 0.6], file_label=f"coherence_{node1}_{node2}"
        )

    def _get_simulated_power(self, node):
        """
        Retrieves the simulated power spectrum for a given node.

        Parameters
        ----------
        node : str
            The name of the brain region for which to retrieve the simulated power spectrum.

        Returns
        -------
        tuple
            A tuple containing:
            - frequencies (numpy.ndarray): The array of frequencies.
            - powers (dict): A dictionary of simulated power spectra, keyed by simulation name.
        """

        simulations = self.simulation_data_extractor.simulations_power_per_node[node]
        frequencies, powers = zip(*[simulations[key] for key in self.simulation_data_extractor.simulation_names])
        return frequencies[0], dict(zip(self.simulation_data_extractor.simulation_names, powers))

    def _get_simulated_coherences(self, node1, node2):
        """
        Computes the simulated coherence between a pair of nodes for each simulation using
        :py:meth:`src.simulation.coherence_computer.CoherenceComputer.compute_coherence_matched` .

        Parameters
        ----------
        node1 : str
            The name of the first brain region.
        node2 : str
            The name of the second brain region.

        Returns
        -------
        tuple
            A tuple containing:
            - frequencies (numpy.ndarray): The array of frequencies for coherence.
            - coherences (dict): A dictionary of simulated coherence values, keyed by simulation name.
        """

        simulations1 = self.simulation_data_extractor.simulations_data_per_node[node1]
        simulations2 = self.simulation_data_extractor.simulations_data_per_node[node2]

        freq_coh_sim = None
        coherences = {}
        keys = self.simulation_data_extractor.simulation_names

        for key in keys:
            time_series1 = simulations1[key]
            time_series2 = simulations2[key]

            sample_rate = self.simulation_data_extractor.sample_rates[key]
            coherence_computer_sim = CoherenceComputer(fs=sample_rate)

            freq_coh_sim, coh = coherence_computer_sim.compute_coherence_matched(time_series1, time_series2)
            coherences[key] = coh

        return freq_coh_sim, coherences

    def _plot_metric(
            self,
            title,
            sim_frequencies, sim_data,
            fig=None, ax=None, show_legend=True, y_label=None, xlim=None, ylim=None, file_label=None
    ):
        """
        Plots a metric (e.g., coherence or power) of data
        using :py:meth:`_plot_simulated_data`.

        Parameters
        ----------
        title : str
            The title of the plot.
        sim_frequencies : numpy.ndarray
            The array of frequencies for the simulated data.
        sim_data : dict
            The simulated data (e.g., power or coherence) to plot, keyed by simulation name.
        fig : matplotlib.figure.Figure, optional
            The figure object for plotting (default is None).
        ax : matplotlib.axes.Axes, optional
            The axis object for plotting (default is None).
        show_legend : bool, optional
            If True, shows the legend on the plot (default is True).
        y_label : str, optional
            The label for the y-axis (default is None).
        xlim : list, optional
            The x-axis limits for the plot (default is None).
        ylim : list, optional
            The y-axis limits for the plot (default is None).
        file_label : str, optional
            The file name label for saving the plot (default is None).
        """

        independent = fig is None or ax is None
        if independent:
            fig, ax = plt.subplots(figsize=(PLOT_SIZE * 2, PLOT_SIZE))

        self._plot_simulated_data(ax, sim_frequencies, sim_data)

        ax.set_title(title)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        # ax.set_ylabel(y_label)
        ax.grid(which='both')

        if show_legend:
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles=handles, loc='upper right')

        if independent:
            # plt.show()
            path = paths.plots_path / f"{file_label}_{title}.{PLOT_FORMAT}"
            fig.savefig(path)

    @staticmethod
    def _plot_simulated_data(ax, frequencies, data):
        """
        Plots the simulated EEG data on a given axis.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axis object for plotting.
        frequencies : numpy.ndarray
            The array of frequencies for the simulated data.
        data : dict
            The simulated data (e.g., power or coherence) to plot, keyed by simulation name.
        """

        for i, (name, d) in enumerate(data.items()):
            ax.plot(frequencies, d, label=name, color=COLORS[i], alpha=1.0)

    @staticmethod
    def _get_ax(ax, cols, i):
        """
        Helper function to get the appropriate subplot axis.

        Parameters
        ----------
        ax : numpy.ndarray
            The array of axis objects for subplots.
        cols : int
            The number of columns in the subplot grid.
        i : int
            The index of the current plot.

        Returns
        -------
        matplotlib.axes.Axes
            The appropriate axis object for the current subplot.
        """

        if ax is None:
            return None
        return ax[i // cols, i % cols]
