macro_eeg_model
===============

.. py:module:: macro_eeg_model


Subpackages
-----------

.. toctree::
   :maxdepth: 1

   /autoapi/macro_eeg_model/config/index
   /autoapi/macro_eeg_model/data_prep/index
   /autoapi/macro_eeg_model/evaluation/index
   /autoapi/macro_eeg_model/simulation/index
   /autoapi/macro_eeg_model/utils/index


Classes
-------

.. autoapisummary::

   macro_eeg_model.GlobalSimulation
   macro_eeg_model.Evaluator


Functions
---------

.. autoapisummary::

   macro_eeg_model.get_simulate_config


Package Contents
----------------

.. py:function:: get_simulate_config()

   First parses command line arguments using :py:func:`get_parsed_args`, then
   creates an instance of the :py:class:`src.config.model_config.ModelConfig`
   class based on the parsed arguments.

   :returns: A tuple containing:

             - config (ModelConfig): The configuration object for the simulation.
             - model_name (str): The name of the model as specified in the command line arguments.
             - n (int): The number of simulations to run, as specified in the command line arguments.
   :rtype: tuple


.. py:class:: GlobalSimulation(config)

   A class responsible for orchestrating the entire simulation process.
   It integrates the development of a stationary model, the simulation of EEG data, data processing, and data analysis.

   The class uses:
   - :py:class:`src.simulation.stationary_model_developer.StationaryModelDeveloper` to create a stationary model from the provided configuration.
   - :py:class:`src.simulation.simulator.Simulator` to generate synthetic EEG data based on the model.
   - :py:class:`src.simulation.data_processor.DataProcessor` to filter and segment the simulated data.
   - :py:class:`src.simulation.eeg_analyzer.EEGAnalyzer` to calculate and plot the power spectrum of the EEG data.
   - :py:class:`src.simulation.simulation_info.SimulationInfo` to save the simulation results.


   .. py:method:: __init__(config)

      Initializes the GlobalSimulation class with the provided configuration.

      :param config: The configuration object containing parameters for the simulation
                     (instance of the :py:class:`src.config.model_config.ModelConfig` class).
      :type config: ModelConfig



   .. py:method:: run(save_data=False, make_plots=False, verbose=False, simulation_name=None)

      Runs the global simulation process, including model development, data simulation, processing, analysis, and optional saving/plotting.

      :param save_data: If True, saves the simulation results (default is False).
      :type save_data: bool, optional
      :param make_plots: If True, generates and saves plots of the connectivity and power spectrum (default is False).
      :type make_plots: bool, optional
      :param verbose: If True, displays progress bars and detailed information during the simulation (default is False).
      :type verbose: bool, optional
      :param simulation_name: The name of the simulation, used for saving the results (default is None).
      :type simulation_name: str, optional

      :returns: A tuple containing:

                - simulation_data (numpy.ndarray): The simulated EEG data.
                - frequencies (numpy.ndarray): The array of frequencies corresponding to the power spectrum.
                - power (numpy.ndarray): The power spectrum of the simulated EEG data.
      :rtype: tuple



.. py:class:: Evaluator

   A class responsible for evaluating simulated EEG data .
   It computes metrics such as coherence and power spectra across different brain regions (nodes).

   .. attribute:: frequencies

      The frequency range for evaluating the data ([0, 30] Hz).

      :type: list

   .. attribute:: simulation_data_extractor

      An instance of the :py:class:`src.evaluation.simulation_data_extractor.SimulationDataExtractor` class
      used to extract and process simulated EEG data.

      :type: SimulationDataExtractor


   .. py:method:: __init__()

      Initializes the Evaluator class, setting up the frequency range and loading real and simulated data.



   .. py:method:: evaluate(plot_overview=True)

      Evaluates and compares the coherence and power metrics
      using :py:meth:`_evaluate_metric`.

      :param plot_overview: If True, generates overview plots for the evaluated metrics;
                            if False, generates individual plots for (pairs of) brain regions.
                            (default is True).
      :type plot_overview: bool, optional



   .. py:method:: _evaluate_metric(evaluation_func, desc, plot_overview, rows, cols, save_file_name)

      A helper function to evaluate a specific metric (e.g., coherence or power) across nodes or node pairs.

      :param evaluation_func: The function to evaluate the metric
                              (:py:meth:`_evaluate_coherence_node_pair` or :py:meth:`_evaluate_power_node`).
      :type evaluation_func: function
      :param desc: The description for the tqdm progress bar.
      :type desc: str
      :param plot_overview: If True, generates overview plots for the evaluated metrics;
                            if False, generates individual plots for (pairs of) brain regions.
      :type plot_overview: bool
      :param rows: The number of rows in the overview plot.
      :type rows: int
      :param cols: The number of columns in the overview plot.
      :type cols: int
      :param save_file_name: The file name for saving the overview plot.
      :type save_file_name: str



   .. py:method:: _get_nodes(pairwise=False)

      Generates nodes or node pairs for evaluation.

      :param pairwise: If True, generates pairs of nodes (for coherence evaluation),
                       otherwise generates individual nodes (for power evaluation)
                       (default is False).
      :type pairwise: bool, optional

      :returns: A tuple containing one or two nodes, depending on the value of `pairwise`.
      :rtype: tuple



   .. py:method:: _evaluate_peaks(node, fig=None, ax=None, show_legend=True)

      Evaluates the presence of alpha peaks (using :py:meth:`_get_peaks`)
      and plots (using :py:meth:`_plot_metric`) detrended power spectrum for a given node.

      :param node: The name of the brain region to evaluate.
      :type node: str
      :param fig: The figure object for plotting (default is None).
      :type fig: matplotlib.figure.Figure, optional
      :param ax: The axis object for plotting (default is None).
      :type ax: matplotlib.axes.Axes, optional
      :param show_legend: If True, shows the legend on the plot (default is True).
      :type show_legend: bool, optional



   .. py:method:: _evaluate_fooof(node, fig=None, ax=None, show_legend=True)

      Evaluates the presence of peaks (using :py:meth:`_get_fooof_peaks`)
      and plots (using :py:meth:`_plot_metric`) the peaks.

      :param node: The name of the brain region to evaluate.
      :type node: str
      :param fig: The figure object for plotting (default is None).
      :type fig: matplotlib.figure.Figure, optional
      :param ax: The axis object for plotting (default is None).
      :type ax: matplotlib.axes.Axes, optional
      :param show_legend: If True, shows the legend on the plot (default is True).
      :type show_legend: bool, optional



   .. py:method:: _evaluate_power_node(node, fig=None, ax=None, show_legend=True)

      Evaluates (using :py:meth:`_get_simulated_power`)
      and plots (using :py:meth:`_plot_metric`) the power spectrum for a given node.

      :param node: The name of the brain region to evaluate.
      :type node: str
      :param fig: The figure object for plotting (default is None).
      :type fig: matplotlib.figure.Figure, optional
      :param ax: The axis object for plotting (default is None).
      :type ax: matplotlib.axes.Axes, optional
      :param show_legend: If True, shows the legend on the plot (default is True).
      :type show_legend: bool, optional



   .. py:method:: _evaluate_coherence_node_pair(node1, node2, fig=None, ax=None, show_legend=True)

      Evaluates (using :py:meth:`_get_simulated_coherences`)
      and plots (using :py:meth:`_plot_metric`)
      the coherence between a pair of nodes.

      :param node1: The name of the first brain region (node).
      :type node1: str
      :param node2: The name of the second brain region (node).
      :type node2: str
      :param fig: The figure object for plotting (default is None).
      :type fig: matplotlib.figure.Figure, optional
      :param ax: The axis object for plotting (default is None).
      :type ax: matplotlib.axes.Axes, optional
      :param show_legend: If True, shows the legend on the plot (default is True).
      :type show_legend: bool, optional



   .. py:method:: _get_fooof_peaks(node)

      Computes the peaks in the power spectrum for a given node using :py:class:`FooofTester`.

      :param node: The name of the brain region for which to compute the peaks.
      :type node: str

      :returns: A tuple containing:

                - frequencies (numpy.ndarray): The array of frequencies.
                - all_binary_peaks (dict): A dictionary of binary peaks for each simulation, keyed by simulation name.
      :rtype: tuple



   .. py:method:: _get_peaks(node)

      Computes the peaks in the power spectrum for a given node using :py:class:`PeakTester`.

      :param node: The name of the brain region for which to compute the peaks.
      :type node: str

      :returns: A tuple containing:

                - frequencies (numpy.ndarray): The array of frequencies.
                - powers (dict): A dictionary of simulated power spectra, keyed by simulation name.
                - p_values (dict): A dictionary of p-values for the peak test, keyed by simulation name.
                - test_names (dict): A dictionary of test names for the peak test, keyed by simulation name.
      :rtype: tuple



   .. py:method:: _get_simulated_power(node)

      Retrieves the simulated power spectrum for a given node.

      :param node: The name of the brain region for which to retrieve the simulated power spectrum.
      :type node: str

      :returns: A tuple containing:

                - frequencies (numpy.ndarray): The array of frequencies.
                - powers (dict): A dictionary of simulated power spectra, keyed by simulation name.
      :rtype: tuple



   .. py:method:: _get_simulated_coherences(node1, node2)

      Computes the simulated coherence between a pair of nodes for each simulation using
      :py:meth:`src.simulation.coherence_computer.CoherenceComputer.compute_coherence_matched` .

      :param node1: The name of the first brain region.
      :type node1: str
      :param node2: The name of the second brain region.
      :type node2: str

      :returns: A tuple containing:

                - frequencies (numpy.ndarray): The array of frequencies for coherence.
                - coherences (dict): A dictionary of simulated coherence values, keyed by simulation name.
      :rtype: tuple



   .. py:method:: _plot_metric(title, sim_frequencies, sim_data, plot_type='line', fig=None, ax=None, show_legend=True, y_label=None, xlim=None, ylim=None, file_label=None, label_addons=None)

      Plots a metric (e.g., coherence or power) of data
      using :py:meth:`_plot_simulated_data`.

      :param title: The title of the plot.
      :type title: str
      :param sim_frequencies: The array of frequencies for the simulated data.
      :type sim_frequencies: numpy.ndarray
      :param sim_data: The simulated data (e.g., power or coherence) to plot, keyed by simulation name.
      :type sim_data: dict
      :param plot_type: The type of plot to create (default is "line"). Currently, "line" and "scatter" are supported.
      :type plot_type: str, optional
      :param fig: The figure object for plotting (default is None).
      :type fig: matplotlib.figure.Figure, optional
      :param ax: The axis object for plotting (default is None).
      :type ax: matplotlib.axes.Axes, optional
      :param show_legend: If True, shows the legend on the plot (default is True).
      :type show_legend: bool, optional
      :param y_label: The label for the y-axis (default is None).
      :type y_label: str, optional
      :param xlim: The x-axis limits for the plot (default is None).
      :type xlim: list, optional
      :param ylim: The y-axis limits for the plot (default is None).
      :type ylim: list, optional
      :param file_label: The file name label for saving the plot (default is None).
      :type file_label: str, optional
      :param label_addons: The dictionary of label addons to append to the name of the data (default is None).
      :type label_addons: dict, optional



   .. py:method:: _plot_simulated_data(ax, frequencies, data, label_addons, plot_type='line')
      :staticmethod:


      Plots the simulated EEG data on a given axis.

      :param ax: The axis object for plotting.
      :type ax: matplotlib.axes.Axes
      :param frequencies: The array of frequencies for the simulated data.
      :type frequencies: numpy.ndarray
      :param data: The simulated data (e.g., power or coherence) to plot, keyed by simulation name.
      :type data: dict
      :param label_addons: The dictionary of label addons to append to the name of the data.
      :type label_addons: dict
      :param plot_type: The type of plot to create (default is "line"). Currently, "line" and "scatter" are supported.
      :type plot_type: str, optional



   .. py:method:: _get_ax(ax, rows, cols, i)
      :staticmethod:


      Helper function to get the appropriate subplot axis.

      :param ax: The array of axis objects for subplots.
      :type ax: numpy.ndarray
      :param rows: The number of rows in the subplot grid.
      :type rows: int
      :param cols: The number of columns in the subplot grid.
      :type cols: int
      :param i: The index of the current plot.
      :type i: int

      :returns: The appropriate axis object for the current subplot.
      :rtype: matplotlib.axes.Axes



