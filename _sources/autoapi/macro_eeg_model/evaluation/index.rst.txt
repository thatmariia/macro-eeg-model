macro_eeg_model.evaluation
==========================

.. py:module:: macro_eeg_model.evaluation

.. autoapi-nested-parse::

   evaluation
   -----------
   This package contains the (power and coherence) evaluation functions to
   compare simulated EEG data against real EEG data.



Submodules
----------

.. toctree::
   :maxdepth: 1

   /autoapi/macro_eeg_model/evaluation/coherence_computer/index
   /autoapi/macro_eeg_model/evaluation/evaluator/index
   /autoapi/macro_eeg_model/evaluation/simulation_data_extractor/index


Classes
-------

.. autoapisummary::

   macro_eeg_model.evaluation.CoherenceComputer
   macro_eeg_model.evaluation.Evaluator
   macro_eeg_model.evaluation.SimulationDataExtractor


Package Contents
----------------

.. py:class:: CoherenceComputer(fs, window_type='hann')

   A class responsible for computing the coherence between signals.

   .. attribute:: fs

      The sampling frequency of the signals.

      :type: int

   .. attribute:: _window_type

      The type of window used for smoothing signals before coherence computation.

      :type: str


   .. py:method:: __init__(fs, window_type='hann')

      Initializes the CoherenceComputer with the given sampling frequency and window type.

      :param fs: The sampling frequency of the signals.
      :type fs: int
      :param window_type: The type of window to apply for smoothing the signals before coherence computation (default is 'hann').
      :type window_type: str, optional



   .. py:method:: compute_coherence_matched(sig1, sig2, smooth_signals=True)

      Computes the coherence between two signals using :py:meth:`_compute_coherence`,
      with an option to smooth the signals before computation using :py:meth:`_smooth_signal`.

      :param sig1: The first signal array.
      :type sig1: numpy.ndarray
      :param sig2: The second signal array.
      :type sig2: numpy.ndarray
      :param smooth_signals: If True, applies a smoothing window to the signals before computing coherence (default is True).
      :type smooth_signals: bool, optional

      :returns: A tuple containing:

                - positive_freqs (numpy.ndarray): The array of positive frequency values.
                - positive_coherence (numpy.ndarray): The coherence values corresponding to the positive frequencies.
      :rtype: tuple

      :raises AssertionError: If the two signals do not have the same shape.



   .. py:method:: _compute_coherence(sig1, sig2)

      Computes the coherence between two signals using their cross-spectrum and power spectra.

      :param sig1: The first signal array with shape (nr_epochs, n_samples).
      :type sig1: numpy.ndarray
      :param sig2: The second signal array with the same shape as `sig1`.
      :type sig2: numpy.ndarray

      :returns: A tuple containing:

                - positive_freqs (numpy.ndarray): The array of positive frequency values.
                - positive_coherence (numpy.ndarray): The coherence values corresponding to the positive frequencies.
      :rtype: tuple



   .. py:method:: _smooth_signal(signal)

      Applies a smoothing window to a signal.

      :param signal: The input signal array to be smoothed.
      :type signal: numpy.ndarray

      :returns: The smoothed signal.
      :rtype: numpy.ndarray



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



   .. py:method:: _plot_metric(title, sim_frequencies, sim_data, fig=None, ax=None, show_legend=True, y_label=None, xlim=None, ylim=None, file_label=None, label_addons=None)

      Plots a metric (e.g., coherence or power) of data
      using :py:meth:`_plot_simulated_data`.

      :param title: The title of the plot.
      :type title: str
      :param sim_frequencies: The array of frequencies for the simulated data.
      :type sim_frequencies: numpy.ndarray
      :param sim_data: The simulated data (e.g., power or coherence) to plot, keyed by simulation name.
      :type sim_data: dict
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



   .. py:method:: _plot_simulated_data(ax, frequencies, data, label_addons)
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



.. py:class:: SimulationDataExtractor

   The SimulationDataExtractor class is responsible for extracting and processing simulation data.
   It organizes the data by nodes and simulations, allowing for easy access to both raw and processed data.

   .. attribute:: nodes

      An array of node names used in the simulations.

      :type: numpy.ndarray

   .. attribute:: simulation_names

      A list of simulation names.

      :type: list

   .. attribute:: sample_rates

      A dictionary mapping simulation names to their corresponding sample rates.

      :type: dict

   .. attribute:: simulations_data_per_node

      A dictionary organizing the processed simulation data by node.

      :type: dict

   .. attribute:: simulations_power_per_node

      A dictionary organizing the processed power spectra by node.

      :type: dict

   .. attribute:: simulations_epoched_power_per_node

      A dictionary organizing the processed epoched power spectra by node.

      :type: dict


   .. py:method:: __init__()

      Initializes the SimulationDataExtractor by loading and processing the simulation data
      using methods from this class.



   .. py:method:: _get_simulations_data_per_node(processed_simulations_data)

      Organizes the processed simulation data by node and then simulation name.

      :param processed_simulations_data: The dictionary containing processed simulation data organized by simulation name and then node.
      :type processed_simulations_data: dict

      :returns: A dictionary organizing the simulation data by node and then simulation name.
      :rtype: dict



   .. py:method:: _get_simulations_epoched_power_per_node(processed_simulations_epoched_power)

      Organizes the processed epoched power spectra by node and then simulation name.

      :param processed_simulations_epoched_power: The dictionary containing processed epoched power spectra organized by simulation name and then node.
      :type processed_simulations_epoched_power: dict

      :returns: A dictionary organizing the epoched power spectra by node and then simulation name.
      :rtype: dict



   .. py:method:: _get_processed_simulations_epoched_power(simulations_info, epoch_len=1000)

      Processes and organizes the epoched power spectra data by simulation name and then node.

      :param simulations_info: A dictionary containing simulation information objects.
      :type simulations_info: dict
      :param epoch_len: The length of each epoch in milliseconds (default is 1000).
      :type epoch_len: int, optional

      :returns: A dictionary organizing the processed epoched power spectra data by simulation name and then node.
      :rtype: dict



   .. py:method:: _get_simulations_power_per_node(processed_simulations_power)

      Organizes the processed power spectra by node and then simulation name.

      :param processed_simulations_power: The dictionary containing processed power spectra organized by simulation name and then node.
      :type processed_simulations_power: dict

      :returns: A dictionary organizing the power spectra by node and then simulation name.
      :rtype: dict



   .. py:method:: _get_processed_simulations_power(simulations_info)

      Processes and organizes the power spectra data by simulation name and then node.

      :param simulations_info: A dictionary containing simulation information objects.
      :type simulations_info: dict

      :returns: A dictionary organizing the processed power spectra data by simulation name and then node.
      :rtype: dict



   .. py:method:: _get_processed_simulations_data(simulations_info)

      Processes and organizes the raw simulation data by simulation name and then node.

      :param simulations_info: A dictionary containing simulation information objects.
      :type simulations_info: dict

      :returns: A dictionary organizing the processed simulation data by simulation name and then node.
      :rtype: dict



   .. py:method:: _get_simulations_info()

      Loads simulation information from saved files in the directories within the
      output path (see :py:class:`src.utils.paths.Paths`) using
      :py:meth:`src.simulation.simulation_info.SimulationInfo.load_simulation_info`.
      and checks for consistency in node names.

      :returns: A tuple containing:

                - simulations_info (dict): A dictionary of SimulationInfo objects keyed by simulation name.
                - sample_rates (dict): A dictionary of sample rates keyed by simulation name.
      :rtype: tuple

      :raises AssertionError: If the nodes in any simulation do not match the expected node names.



   .. py:method:: _get_surface_nodes(nodes)
      :staticmethod:


      Returns the surface nodes from the given list of nodes.
      Currently, the surface nodes are all nodes except the thalamus.

      :param nodes: A list of node names.
      :type nodes: list

      :returns: A list of surface node names.
      :rtype: list



