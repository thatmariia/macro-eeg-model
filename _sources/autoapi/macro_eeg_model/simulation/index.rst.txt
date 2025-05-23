macro_eeg_model.simulation
==========================

.. py:module:: macro_eeg_model.simulation

.. autoapi-nested-parse::

   simulation
   -----------
   This package contains the simulation scripts for the model simulation.



Submodules
----------

.. toctree::
   :maxdepth: 1

   /autoapi/macro_eeg_model/simulation/data_processor/index
   /autoapi/macro_eeg_model/simulation/delay_calculator/index
   /autoapi/macro_eeg_model/simulation/distributions/index
   /autoapi/macro_eeg_model/simulation/eeg_analyzer/index
   /autoapi/macro_eeg_model/simulation/global_simulation/index
   /autoapi/macro_eeg_model/simulation/simulation_info/index
   /autoapi/macro_eeg_model/simulation/simulator/index
   /autoapi/macro_eeg_model/simulation/stationary_model_developer/index


Classes
-------

.. autoapisummary::

   macro_eeg_model.simulation.DataProcessor
   macro_eeg_model.simulation.DelayCalculator
   macro_eeg_model.simulation.LagDistributions
   macro_eeg_model.simulation.DistributionFactory
   macro_eeg_model.simulation.InverseGEV
   macro_eeg_model.simulation.InverseGEVSum
   macro_eeg_model.simulation.EEGAnalyzer
   macro_eeg_model.simulation.GlobalSimulation
   macro_eeg_model.simulation.SimulationInfo
   macro_eeg_model.simulation.Simulator
   macro_eeg_model.simulation.StationaryModelDeveloper


Package Contents
----------------

.. py:class:: DataProcessor

   A class responsible for processing EEG data by filtering and segmenting it.


   .. py:method:: filter_data(data, sample_rate, pass_frequency, stop_frequency)
      :staticmethod:


      Filters the data using a high-pass Butterworth filter
      based on the specified passband and stopband frequencies.

      :param data: The input data to be filtered (a 2D array where rows represent time points and columns represent channels/nodes).
      :type data: numpy.ndarray
      :param sample_rate: The sample rate of the data in Hz.
      :type sample_rate: int
      :param pass_frequency: The passband edge frequency in Hz.
      :type pass_frequency: float
      :param stop_frequency: The stopband edge frequency in Hz.
      :type stop_frequency: float

      :returns: The filtered data with the same shape as the input data.
      :rtype: numpy.ndarray

      :raises AssertionError: If the frequency values are invalid.



   .. py:method:: segment_data(data, sample_rate, nr_nodes)
      :staticmethod:


      Segments the data into epochs of 1 second each, evenly dividing the data based on the sample rate.

      :param data: The input data to be segmented (a 2D array where rows represent time points and columns represent channels/nodes).
      :type data: numpy.ndarray
      :param sample_rate: The sample rate of the data in Hz.
      :type sample_rate: int
      :param nr_nodes: The number of nodes (channels) in the data.
      :type nr_nodes: int

      :returns: A 3D array where each slice along the third dimension represents a 1-second epoch of the data.
                The shape of the array is (t_samples, nr_nodes, nr_epochs), where t_samples is the number of samples per second.
      :rtype: numpy.ndarray



.. py:class:: DelayCalculator(shape_param, scale_param, location_param, truncation_percentile)

   A class to calculate delay distributions based on distance and various statistical parameters.
   The delay is modeled using inverse generalized extreme value (GEV) distributions.

   .. attribute:: _shape_param

      The shape parameter (xi) for the GEV distribution.

      :type: float

   .. attribute:: _scale_param

      The scale parameter (sigma) for the GEV distribution.

      :type: float

   .. attribute:: _location_param

      The location parameter (mu) for the GEV distribution.

      :type: float

   .. attribute:: _truncation_percentile

      The percentile at which to truncate the resulting delay distribution.

      :type: float

   .. attribute:: _velocity_factor

      A constant velocity factor (= 6) used to calculate the scale coefficient from the distance.
      Expressed in meters per second per micron diameter.

      :type: float


   .. py:method:: __init__(shape_param, scale_param, location_param, truncation_percentile)

      Initializes the DelayCalculator with specified parameters for the GEV distribution
      and the truncation percentile.

      :param shape_param: The shape parameter (xi) for the GEV distribution.
      :type shape_param: float
      :param scale_param: The scale parameter (sigma) for the GEV distribution.
      :type scale_param: float
      :param location_param: The location parameter (mu) for the GEV distribution.
      :type location_param: float
      :param truncation_percentile: The percentile at which to truncate the resulting delay distribution. Must be in the range [0, 1).
      :type truncation_percentile: float



   .. py:method:: get_delays_distribution(tempx, distance)

      Generates a probability density function (PDF) for delays using inverse GEV distributions.
      The distributions are also scaled with parameter computed by :py:meth:`_calculate_scale_coefficient`.

      Depending on whether the distance is a single value or a tuple (in case of a relay station),
      it either sums inverse GEV distributions or uses a single inverse GEV distribution
      (see :py:class:`src.simulation.distributions.LagDistributions` and
      :py:class:`src.simulation.distributions.DistributionFactory`).
      It then truncates the resulting PDF using :py:meth:`_truncate_result`.

      :param tempx: The array of time points (x-axis) over which to calculate the delay distribution.
      :type tempx: numpy.ndarray
      :param distance: The distance(s) over which to calculate the delay. If a tuple, the method will use
                       the sum of two inverse GEV distributions.
      :type distance: float or tuple

      :returns: The truncated probability density function (PDF) representing the delay distribution.
      :rtype: numpy.ndarray

      :raises AssertionError: If the distribution cannot be created.



   .. py:method:: _calculate_scale_coefficient(distance)

      Calculates the scale coefficient for the GEV distribution based on the given distance.

      :param distance: The distance for which to calculate the scale coefficient.
      :type distance: float

      :returns: The scale coefficient used in the GEV distribution.
      :rtype: float



   .. py:method:: _truncate_result(tempx, result)

      Truncates the PDF by setting values beyond a certain index to zero, based on the cumulative
      distribution function (CDF) and the truncation percentile.

      :param tempx: The array of time points (x-axis) corresponding to the PDF.
      :type tempx: numpy.ndarray
      :param result: The PDF to be truncated.
      :type result: numpy.ndarray

      :returns: The truncated PDF.
      :rtype: numpy.ndarray

      :raises AssertionError: If the truncation percentile is outside the valid range [0, 1).



.. py:class:: LagDistributions(*args, **kwds)

   Bases: :py:obj:`enum.Enum`


   An enumeration for different types of lag distributions.

   .. attribute:: INVERSE_GEV

      Represents an inverse generalized extreme value (GEV) distribution.

      :type: str

   .. attribute:: INVERSE_GEV_SUM

      Represents the sum of two inverse GEV distributions.

      :type: str


   .. py:method:: __init__(*args, **kwds)


.. py:class:: DistributionFactory

   A factory class responsible for creating different types of distributions based on the provided type.


   .. py:method:: get_distribution(distribution_type, **kwargs)
      :staticmethod:


      Creates and returns a distribution object based on the specified type.

      :param distribution_type: The type of distribution to create (e.g., INVERSE_GEV, INVERSE_GEV_SUM).
      :type distribution_type: LagDistributions
      :param kwargs: The parameters required to initialize the distribution.
      :type kwargs: dict

      :returns: An instance of a distribution class (e.g., :py:class:`InverseGEV`, :py:class:`InverseGEVSum`).
      :rtype: rv_continuous

      :raises ValueError: If an unknown distribution type is provided.



.. py:class:: InverseGEV(lmbd, mu, sigma, xi, *args, **kwargs)

   Bases: :py:obj:`scipy.stats.rv_continuous`


   A class representing the inverse generalized extreme value (GEV) distribution.

   This class extends `scipy.stats.rv_continuous` to model the inverse GEV distribution.

   .. attribute:: lmbd

      A scaling parameter applied to the distribution.

      :type: float

   .. attribute:: mu

      The location parameter of the GEV distribution.

      :type: float

   .. attribute:: sigma

      The scale parameter of the GEV distribution.

      :type: float

   .. attribute:: xi

      The shape parameter of the GEV distribution.

      :type: float


   .. py:method:: __init__(lmbd, mu, sigma, xi, *args, **kwargs)

      Initializes the InverseGEV distribution with the specified parameters.

      :param lmbd: A scaling parameter applied to the distribution.
      :type lmbd: float
      :param mu: The location parameter of the GEV distribution.
      :type mu: float
      :param sigma: The scale parameter of the GEV distribution.
      :type sigma: float
      :param xi: The shape parameter of the GEV distribution.
      :type xi: float



   .. py:method:: _argcheck(*args)

      Validates the distribution parameters.

      :returns: True if the parameters are valid, False otherwise.
      :rtype: bool



   .. py:method:: _cdf(x, *args)

      Calculates the cumulative distribution function (CDF) for the inverse GEV.

      :param x: The quantiles at which to evaluate the CDF.
      :type x: array_like

      :returns: The CDF evaluated at the given quantiles.
      :rtype: array_like



   .. py:method:: _pdf(x, *args)

      Calculates the probability density function (PDF) for the inverse GEV.

      :param x: The quantiles at which to evaluate the PDF.
      :type x: array_like

      :returns: The PDF evaluated at the given quantiles.
      :rtype: array_like



   .. py:method:: _ppf(q, *args)

      Calculates the percent point function (PPF), also known as the quantile function, for the inverse GEV.

      :param q: The quantiles for which to evaluate the PPF.
      :type q: array_like

      :returns: The PPF evaluated at the given quantiles.
      :rtype: array_like



   .. py:method:: _rvs(*args, size=None, random_state=None)

      Generates random variates from the inverse GEV distribution.

      :param size: The number of random variates to generate.
      :type size: int or tuple of ints, optional
      :param random_state: A random state instance for reproducibility.
      :type random_state: np.random.RandomState, optional

      :returns: The generated random variates.
      :rtype: array_like



.. py:class:: InverseGEVSum(lmbd1, lmbd2, mu, sigma, xi, *args, **kwargs)

   Bases: :py:obj:`scipy.stats.rv_continuous`


   A class representing the sum of two inverse GEV distributions.

   This class extends `scipy.stats.rv_continuous` to model the sum of two inverse GEV distributions.
   The sum is approximated using a kernel density estimate (KDE) of the sum of samples from the two distributions.

   .. attribute:: lmbd1

      A scaling parameter for the first inverse GEV distribution.

      :type: float

   .. attribute:: lmbd2

      A scaling parameter for the second inverse GEV distribution.

      :type: float

   .. attribute:: mu

      The location parameter of the GEV distributions.

      :type: float

   .. attribute:: sigma

      The scale parameter of the GEV distributions.

      :type: float

   .. attribute:: xi

      The shape parameter of the GEV distributions.

      :type: float

   .. attribute:: _kde

      The kernel density estimate of the sum of the two inverse GEV distributions.

      :type: KernelDensity


   .. py:method:: __init__(lmbd1, lmbd2, mu, sigma, xi, *args, **kwargs)

      Initializes the InverseGEVSum distribution with the specified parameters.

      :param lmbd1: A scaling parameter for the first inverse GEV distribution.
      :type lmbd1: float
      :param lmbd2: A scaling parameter for the second inverse GEV distribution.
      :type lmbd2: float
      :param mu: The location parameter of the GEV distributions.
      :type mu: float
      :param sigma: The scale parameter of the GEV distributions.
      :type sigma: float
      :param xi: The shape parameter of the GEV distributions.
      :type xi: float



   .. py:method:: _get_kde()

      Generates a kernel density estimate (KDE) for the sum of two inverse GEV distributions.

      :returns: A KDE fitted to the sum of samples from the two inverse GEV distributions.
      :rtype: KernelDensity



   .. py:method:: _argcheck(*args)

      Validates the distribution parameters.

      :returns: True if the parameters are valid, False otherwise.
      :rtype: bool



   .. py:method:: _pdf(x, *args)

      Evaluates the kernel density estimate (KDE) at the given quantiles to
      approximate the probability density function (PDF) for the sum of inverse GEVs.

      :param x: The quantiles at which to evaluate the PDF.
      :type x: array_like

      :returns: The PDF evaluated at the given quantiles.
      :rtype: array_like



.. py:class:: EEGAnalyzer

   The EEGAnalyzer class is responsible computing the power spectrum of EEG data.


   .. py:method:: calculate_power(data, sample_rate)
      :staticmethod:


      Applies the Fast Fourier Transform (FFT) to the EEG data to calculate the power spectrum.
      It returns the frequencies and the average power spectrum across epochs/samples per second.

      :param data: The EEG data to be analyzed (a 3D array with dimensions (time, nodes, epochs)).
      :type data: numpy.ndarray
      :param sample_rate: The sample rate of the EEG data in Hz.
      :type sample_rate: int

      :returns: A tuple containing:

                - frequencies (numpy.ndarray): The array of frequencies corresponding to the power spectrum.
                - power (numpy.ndarray): The calculated power spectrum for each frequency and node.
      :rtype: tuple

      :raises ValueError: If the user-defined frequencies are outside the valid range determined by the Nyquist frequency.



   .. py:method:: plot_power(frequencies, power, nodes, plots_dir)
      :staticmethod:


      Visualizes the power spectrum of the EEG data (for each node/channel) as a line plot.

      :param frequencies: The array of frequencies corresponding to the power spectrum.
      :type frequencies: numpy.ndarray
      :param power: The calculated power spectrum for each frequency and node.
      :type power: numpy.ndarray
      :param nodes: The list of node/channel names corresponding to the data.
      :type nodes: list[str]
      :param plots_dir: The directory where the plots are saved.
      :type plots_dir: pathlib.Path

      :raises AssertionError: If the plots directory does not exist.



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



.. py:class:: SimulationInfo(output_dir, nodes=None, distances=None, connectivity_weights=None, sample_rate=None, lag_connectivity_weights=None, simulation_data=None, frequencies=None, power=None)

   A class responsible for storing and retrieving information about a simulation.
   It handles saving and loading the data related to a simulation, such as nodes, distances, connectivity weights, and results.

   .. attribute:: nodes

      The array of nodes used in the simulation.

      :type: numpy.ndarray

   .. attribute:: distances

      The distance matrix between nodes used in the simulation.

      :type: numpy.ndarray

   .. attribute:: connectivity_weights

      The connectivity weights matrix between nodes.

      :type: numpy.ndarray

   .. attribute:: sample_rate

      The sample rate of the simulation in Hz.

      :type: int

   .. attribute:: lag_connectivity_weights

      The lagged connectivity weights matrix used in the VAR model.

      :type: numpy.ndarray

   .. attribute:: simulation_data

      The simulated EEG data.

      :type: numpy.ndarray

   .. attribute:: frequencies

      The array of frequencies corresponding to the power spectrum.

      :type: numpy.ndarray

   .. attribute:: power

      The power spectrum calculated from the simulation data.

      :type: numpy.ndarray

   .. attribute:: _output_dir

      The directory path where simulation results are saved.

      :type: pathlib.Path


   .. py:method:: __init__(output_dir, nodes=None, distances=None, connectivity_weights=None, sample_rate=None, lag_connectivity_weights=None, simulation_data=None, frequencies=None, power=None)

      Initializes the SimulationInfo class with the provided simulation parameters and data.

      :param output_dir: The path to the output directory where simulation results are saved.
      :type output_dir: pathlib.Path
      :param nodes: The array of nodes used in the simulation.
      :type nodes: numpy.ndarray, optional
      :param distances: The distance matrix between nodes used in the simulation.
      :type distances: numpy.ndarray, optional
      :param connectivity_weights: The connectivity weights matrix between nodes.
      :type connectivity_weights: numpy.ndarray, optional
      :param sample_rate: The sample rate of the simulation in Hz.
      :type sample_rate: int, optional
      :param lag_connectivity_weights: The lagged connectivity weights matrix used in the VAR model.
      :type lag_connectivity_weights: numpy.ndarray, optional
      :param simulation_data: The simulated EEG data.
      :type simulation_data: numpy.ndarray, optional
      :param frequencies: The array of frequencies corresponding to the power spectrum.
      :type frequencies: numpy.ndarray, optional
      :param power: The power spectrum calculated from the simulation data.
      :type power: numpy.ndarray, optional

      :raises AssertionError: If the output directory does not exist.



   .. py:method:: save_simulation_info()

      Saves the simulation data to the output directory as .npy files.
      The data includes nodes, distances, connectivity weights,
      sample rate, lag connectivity weights, simulation data,
      frequencies, and power spectrum.



   .. py:method:: load_simulation_info()

      Loads all the relevant data of the simulation from the output directory
      and assigns them to the corresponding attributes of the class.

      :raises FileNotFoundError: If any of the required files are not found in the output directory.



.. py:class:: Simulator(lag_connectivity_weights, sample_rate, nr_lags, nr_nodes, t_secs, t_burnit, noise_color, std_noise)

   The Simulator class is responsible for simulating EEG data using a vector autoregression (VAR) model.
   It generates synthetic EEG signals based on the provided lagged connectivity weights, noise characteristics,
   and other simulation parameters.

   .. attribute:: _lag_connectivity_weights

      The lagged connectivity weights matrix used for the VAR model.

      :type: numpy.ndarray

   .. attribute:: _sample_rate

      The sample rate of the simulation in Hz.

      :type: int

   .. attribute:: _nr_lags

      The number of lags (p) in the VAR(p) model.

      :type: int

   .. attribute:: _nr_nodes

      The number of nodes (channels) in the simulation.

      :type: int

   .. attribute:: _t_secs

      The total time of the simulation in seconds.

      :type: int

   .. attribute:: _t_burnit

      The burn-in time for the simulation in seconds.

      :type: int

   .. attribute:: _noise_color

      The color of the noise to be used in the simulation ('white' or 'pink').

      :type: str

   .. attribute:: _std_noise

      The standard deviation of the noise to be used in the simulation.

      :type: float


   .. py:method:: __init__(lag_connectivity_weights, sample_rate, nr_lags, nr_nodes, t_secs, t_burnit, noise_color, std_noise)

      Initializes the Simulator with the provided parameters.

      :param lag_connectivity_weights: The lagged connectivity weights matrix used for the VAR model.
      :type lag_connectivity_weights: numpy.ndarray
      :param sample_rate: The sample rate of the simulation in Hz.
      :type sample_rate: int
      :param nr_lags: The number of lags (p) in the VAR(p) model.
      :type nr_lags: int
      :param nr_nodes: The number of nodes (channels) in the simulation.
      :type nr_nodes: int
      :param t_secs: The total time of the simulation in seconds.
      :type t_secs: int
      :param t_burnit: The burn-in time for the simulation in seconds.
      :type t_burnit: int
      :param noise_color: The color of the noise to be used in the simulation ('white' or 'pink').
      :type noise_color: str
      :param std_noise: The standard deviation of the noise to be used in the simulation.
      :type std_noise: float



   .. py:method:: simulate(verbose=False)

      The simulation generates synthetic EEG signals by applying the VAR model to the provided
      lagged connectivity weights and adding noise.

      :param verbose: If True, displays a progress bar during the simulation (default is False).
      :type verbose: bool, optional

      :returns: A 2D array of shape (samples, nodes) containing the simulated EEG data.
      :rtype: numpy.ndarray

      :raises ValueError: If an invalid noise color is provided.
      :raises AssertionError: If any of the input parameters are invalid (e.g., non-positive values for number of lags, time, or std).



.. py:class:: StationaryModelDeveloper(nr_lags, nr_nodes, nodes, distances, connectivity_weights, sample_rate, delay_calculator)

   A class to develop a stationary vector autoregression (VAR) model from given parameters.

   .. attribute:: _nr_lags

      The number of lags (p) in the VAR(p) model.

      :type: int

   .. attribute:: _nr_nodes

      The number of nodes in the model.

      :type: int

   .. attribute:: _nodes

      The list of node names.

      :type: list[str]

   .. attribute:: _distances

      A matrix containing the distances between nodes.

      :type: numpy.ndarray

   .. attribute:: _connectivity_weights

      The initial connectivity weights between nodes.

      :type: numpy.ndarray

   .. attribute:: _sample_rate

      The sample rate used for the model.

      :type: int

   .. attribute:: _delay_calculator

      An instance of the :py:class:`src.simulation.delay_calculator.DelayCalculator` class used to calculate delay distributions.

      :type: DelayCalculator

   .. attribute:: _tempx

      The array of lag indices.

      :type: numpy.ndarray

   .. attribute:: _delays_x

      The array of delay values based on the sample rate.

      :type: numpy.ndarray


   .. py:method:: __init__(nr_lags, nr_nodes, nodes, distances, connectivity_weights, sample_rate, delay_calculator)

      Initializes the StationaryModelDeveloper with the provided parameters.

      :param nr_lags: The number of lags (p) in the VAR(p) model.
      :type nr_lags: int
      :param nr_nodes: The number of nodes in the model.
      :type nr_nodes: int
      :param nodes: The list of node names.
      :type nodes: list[str]
      :param distances: A matrix containing the distances between nodes.
      :type distances: numpy.ndarray
      :param connectivity_weights: The initial connectivity weights between nodes.
      :type connectivity_weights: numpy.ndarray
      :param sample_rate: The sample rate used for the model.
      :type sample_rate: int
      :param delay_calculator: An instance of the :py:class:`src.simulation.delay_calculator.DelayCalculator` class used to calculate delay distributions.
      :type delay_calculator: DelayCalculator



   .. py:method:: develop(verbose=False)

      Develops a stationary VAR(p) model.

      It calculates the lag connectivity weights using :py:meth:`_calculate_lag_connectivity_weights`,
      and adjusts the overall connectivity weights using :py:meth:`_adjust_connectivity_weights`
      until the model becomes stationary (check with :py:meth:`_is_stationary`).

      :param verbose: If True, displays progress information during the model development (default is False).
      :type verbose: bool, optional

      :returns: The lag connectivity weights matrix for the stationary model.
      :rtype: numpy.ndarray



   .. py:method:: _adjust_connectivity_weights()

      Adjusts the connectivity weights by scaling them down (preserving the relative weights).



   .. py:method:: _is_stationary(lag_connectivity_weights)

      Determines whether the model is stationary.

      It constructs an augmented matrix from the lag connectivity weights and checks
      if all eigenvalues are within the unit circle.

      :param lag_connectivity_weights: The matrix of lag connectivity weights.
      :type lag_connectivity_weights: numpy.ndarray

      :returns: True if the model is stationary (i.e., all eigenvalues are within the unit circle), False otherwise.
      :rtype: bool



   .. py:method:: _calculate_lag_connectivity_weights()

      Computes the connectivity weights for each lag between all pairs of nodes
      using :py:meth:`_get_lag_distribution`.

      :returns: The matrix of lag connectivity weights.
      :rtype: numpy.ndarray



   .. py:method:: _get_lag_distribution(node1, node2)

      Calculates the lag distribution (using :py:attr:_delay_calculator and
      :py:meth:src.simulation.delay_calculator.DelayCalculator.get_delays_distribution)
      between two nodes based on their delays and connectivity weights.
      If the nodes are the same, the distribution is set to zero.

      :param node1: The index of the first node.
      :type node1: int
      :param node2: The index of the second node.
      :type node2: int

      :returns: The lag distribution values, or 0 if the nodes are the same.
      :rtype: numpy.ndarray or int



   .. py:method:: plot_connectivity(lag_connectivity_weights, plots_dir)

      Visualizes the lag connectivity weights between nodes as a line plot,
      showing the relative strength of connections over different delays.

      :param lag_connectivity_weights: The matrix of lag connectivity weights to be plotted.
      :type lag_connectivity_weights: numpy.ndarray
      :param plots_dir: The directory where the plots are saved.
      :type plots_dir: pathlib.Path

      :raises AssertionError: If the plots directory does not exist.



