# standard imports
import sys

# external imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from tqdm import tqdm

# local imports
from evaluation.alphawaves_data_extractor import AlphaWavesDataExtractor
from evaluation.simulation_data_extractor import SimulationDataExtractor
from evaluation.coherence_computer import CoherenceComputer
from utils.plotting_setup import PLOT_SIZE, COLORS, notation, PLOT_FORMAT
from utils.paths import paths


class Evaluator:
    """
    A class responsible for evaluating simulated EEG data against real EEG data from the AlphaWaves dataset.
    It computes and compares metrics such as coherence and power spectra across different brain regions (nodes).

    Attributes
    ----------
    frequencies : list
        The frequency range for evaluating the data ([0, 30] Hz).
    event : str
        The event type for which the real data is extracted (eyes closed condition).
    alpha_waves_data_extractor : AlphaWavesDataExtractor
        An instance of the :py:class:`src.evaluation.alphawaves_data_extractor.AlphaWavesDataExtractor` class
        used to extract and process real EEG data.
    simulation_data_extractor : SimulationDataExtractor
        An instance of the :py:class:`src.evaluation.simulation_data_extractor.SimulationDataExtractor` class
        used to extract and process simulated EEG data.
    _real_avg_time_series : dict
        A dictionary storing the average time series data for each brain region in the real data.
    _real_frequencies : numpy.ndarray
        The frequency values associated with the real power spectrum.
    _real_powers : dict
        A dictionary storing the power spectrum data for each brain region in the real data.
    """

    def __init__(self):
        """
        Initializes the Evaluator class, setting up the frequency range, event type, and loading real and simulated data.
        """

        self.frequencies = (0, 30)
        self.event = "closed"

        self.alpha_waves_data_extractor = AlphaWavesDataExtractor(fs=256, frequencies=self.frequencies)
        self.simulation_data_extractor = SimulationDataExtractor()

        # real
        self._real_avg_time_series = {}
        self._real_frequencies = None
        self._real_powers = {}
        self._extract_real_data()

    def evaluate(self, plot_overview=True):
        """
        Evaluates and compares the coherence and power metrics between simulated and real data
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

    def _extract_real_data(self):
        """
        Extracts the average time series and power spectrum
        for each brain region (node) across all subjects from the AlphaWaves dataset using
        :py:meth:`src.evaluation.alphawaves_data_extractor.AlphaWavesDataExtractor.get_time_series_per_subject_collection_event`.
        """

        for node in self.alpha_waves_data_extractor.channels_collections.keys():
            self._real_avg_time_series[node] = []
            self._real_powers[node] = []

            for subject_id in range(len(self.alpha_waves_data_extractor.subjects)):
                avg_time_series, f, avg_psd = self.alpha_waves_data_extractor.get_time_series_per_subject_collection_event(
                    subject_id=subject_id,
                    channels_collection=self.alpha_waves_data_extractor.channels_collections[node],
                    event=self.event
                )
                self._real_avg_time_series[node].append(avg_time_series)
                self._real_powers[node].append(avg_psd)

                if self._real_frequencies is None:
                    self._real_frequencies = f

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

        nodes = list(self.alpha_waves_data_extractor.channels_collections.keys())
        if pairwise:
            for i, node1 in enumerate(nodes):
                for node2 in nodes[i + 1:]:
                    yield node1, node2
        else:
            for node in nodes:
                yield (node,)

    def _evaluate_power_node(self, node, fig=None, ax=None, show_legend=True):
        """
        Evaluates (using extracted real data :py:meth:`_get_simulated_power` for simulated data)
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

        real_frequencies, real_powers = self._real_frequencies, self._real_powers[node]

        sim_frequencies, sim_powers = self._get_simulated_power(node)

        self._plot_metric(
            f"{notation(node)}",
            real_frequencies, real_powers,
            sim_frequencies, sim_powers,
            fig=fig, ax=ax, show_legend=show_legend if ax is not None else True,
            y_label="Power", xlim=[self.frequencies[0], self.frequencies[1]], ylim=[0, 2e7], file_label=f"power_{node}"
        )

    def _evaluate_coherence_node_pair(self, node1, node2, fig=None, ax=None, show_legend=True):
        """
        Evaluates (using :py:meth:`_get_real_coherences_per_subject` for real data and
        :py:meth:`_get_simulated_coherences` for simulated data)
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

        # real eeg data
        real_frequencies_coherence, real_coherences_per_subject = self._get_real_coherences_per_subject(node1, node2)

        # simulated data
        sim_frequencies_coherence, sim_coherences = self._get_simulated_coherences(node1, node2)

        self._plot_metric(
            f"{notation(node1)} — {notation(node2)}",
            real_frequencies_coherence, real_coherences_per_subject,
            sim_frequencies_coherence, sim_coherences,
            fig=fig, ax=ax, show_legend=show_legend if ax is not None else True,
            y_label="Coherence", xlim=[1, self.frequencies[1]], ylim=[0, 1], file_label=f"coherence_{node1}_{node2}"
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

    def _get_real_coherences_per_subject(self, node1, node2):
        """
        Computes the coherence between a pair of nodes across all subjects in the real dataset
        using :py:meth:`src.simulation.coherence_computer.CoherenceComputer.compute_coherence_matched`.

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
            - coherences (numpy.ndarray): An array of coherence values across all subjects.
        """

        coherence_computer_real = CoherenceComputer(fs=self.alpha_waves_data_extractor.fs)
        coherences = []

        for subject_id in range(len(self.alpha_waves_data_extractor.subjects)):
            avg_time_series1 = self._real_avg_time_series[node1][subject_id]
            avg_time_series2 = self._real_avg_time_series[node2][subject_id]

            freq_coh_real, coh = coherence_computer_real.compute_coherence_matched(avg_time_series1, avg_time_series2)
            coherences.append(coh)

        return freq_coh_real, np.array(coherences)

    def _plot_metric(
            self,
            title,
            real_frequencies, real_data,
            sim_frequencies, sim_data,
            fig=None, ax=None, show_legend=True, y_label=None, xlim=None, ylim=None, file_label=None
    ):
        """
        Plots a comparison of a metric (e.g., coherence or power) between real and simulated data
        using :py:meth:`_plot_real_data` and :py:meth:`_plot_simulated_data`.

        Parameters
        ----------
        title : str
            The title of the plot.
        real_frequencies : numpy.ndarray
            The array of frequencies for the real data.
        real_data : numpy.ndarray
            The real data (e.g., power or coherence) to plot.
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

        self._plot_real_data(ax, real_frequencies, real_data)
        self._plot_simulated_data(ax, sim_frequencies, sim_data)

        ax.set_title(title)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        # ax.set_ylabel(y_label)
        ax.grid(which='both')

        if show_legend:
            handles, labels = ax.get_legend_handles_labels()
            handles = [Line2D([0], [0], color="000000", label="AW subjects", alpha=0.1)] + handles
            ax.legend(handles=handles, loc='upper right')

        if independent:
            # plt.show()
            path = paths.plots_path / f"{file_label}_{title}.{PLOT_FORMAT}"
            fig.savefig(path)

    @staticmethod
    def _plot_real_data(ax, frequencies, data):
        """
        Plots the real EEG data on a given axis.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axis object for plotting.
        frequencies : numpy.ndarray
            The array of frequencies for the real data.
        data : numpy.ndarray
            The real data (e.g., power or coherence) to plot.
        """

        for d in data:
            ax.plot(frequencies, d, color="000000", alpha=0.02)
        mean_data = np.mean(data, axis=0)
        std_data = np.std(data, axis=0)
        ax.fill_between(
            frequencies,
            mean_data - std_data,
            mean_data + std_data,
            color="000000",
            alpha=0.05,
            label="std AW subjects"
        )
        ax.plot(frequencies, mean_data, linestyle="dotted", label="mean AW subjects", color="000000", alpha=0.5)

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
