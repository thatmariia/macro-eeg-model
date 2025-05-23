macro_eeg_model.config
======================

.. py:module:: macro_eeg_model.config

.. autoapi-nested-parse::

   config
   -------
   This package contains the configuration files for the simulation.



Submodules
----------

.. toctree::
   :maxdepth: 1

   /autoapi/macro_eeg_model/config/configs/index
   /autoapi/macro_eeg_model/config/connectivity_model/index
   /autoapi/macro_eeg_model/config/model_config/index
   /autoapi/macro_eeg_model/config/nodes_processor/index
   /autoapi/macro_eeg_model/config/parser/index


Classes
-------

.. autoapisummary::

   macro_eeg_model.config.ConnectivityModel
   macro_eeg_model.config.ModelConfig
   macro_eeg_model.config.NodesProcessor
   macro_eeg_model.config.Parser


Functions
---------

.. autoapisummary::

   macro_eeg_model.config.get_simulate_config
   macro_eeg_model.config.get_parsed_args


Package Contents
----------------

.. py:function:: get_simulate_config()

   First parses command line arguments using :py:func:`get_parsed_args`, then
   creates an instance of the :py:class:`src.config.model_config.ModelConfig`
   class based on the parsed arguments.

   :returns: A tuple containing:

             - config (ModelConfig): The configuration object for the simulation.
             - model_name (str): The name of the model as specified in the command line arguments.
   :rtype: tuple


.. py:function:: get_parsed_args()

   Parses command line arguments using :py:meth:`src.config.parser.Parser.parse_args`.

   :returns: The parsed command line arguments as an `argparse.Namespace` object.
   :rtype: argparse.Namespace


.. py:class:: ConnectivityModel(given_nodes, relay_station)

   A class to model the connectivity between brain nodes. It computes
   distances and connectivity weights between nodes, with optional
   relay stations.

   .. attribute:: nodes

      The processed list of nodes used in the model.

      :type: list

   .. attribute:: nr_nodes

      The total number of nodes in the model.

      :type: int

   .. attribute:: distances

      The matrix of distances between nodes.

      :type: numpy.ndarray

   .. attribute:: connectivity_weights

      The matrix of connectivity weights between nodes.

      :type: numpy.ndarray

   .. attribute:: _given_nodes

      The list of nodes provided for the connectivity model.

      :type: list

   .. attribute:: _relay_station

      The relay station node name, if any.

      :type: str

   .. attribute:: _relay_nodes

      The list of relay nodes derived from the relay station, if applicable.

      :type: list

   .. attribute:: _relay_indices

      The indices of the relay nodes in the connectivity model.

      :type: list

   .. attribute:: _nodes_indices

      A dictionary mapping each node to its corresponding indices.

      :type: dict

   .. attribute:: _avg_counts

      The average counts of connections between nodes.

      :type: numpy.ndarray

   .. attribute:: _avg_fc

      The average functional connectivity between nodes.

      :type: numpy.ndarray

   .. attribute:: _avg_lengths

      The average distances (lengths) between nodes.

      :type: numpy.ndarray

   .. attribute:: _relay_distances

      The dictionary of average distances between nodes and the relay station.

      :type: dict


   .. py:method:: __init__(given_nodes, relay_station)

      Initializes the ConnectivityModel with given nodes and an optional relay station.

      :param given_nodes: The list of nodes to be used in the connectivity model.
      :type given_nodes: list
      :param relay_station: The relay station name (or None).
      :type relay_station: str



   .. py:method:: set_connectivity(custom_connectivity)

      Assigns streamline lengths to the distances matrix (relayed, if applicable) and
      weights to the connectivity matrix based on either
      custom-provided values or default calculations based on functional connectivity (FC).

      The values for a pair of nodes are extracted from :py:meth:`_get_pair_stats`.

      :param custom_connectivity: If True, attempts to load and use custom connectivity weights from
                                  `connectivity_weights.csv` file in the configs path (see :py:class:`src.utils.paths.Paths`).
      :type custom_connectivity: bool

      :raises AssertionError: If the shape of the custom connectivity matrix is incorrect or the matrix has been incorrectly constructed.



   .. py:method:: _get_pair_stats(node1, node2)

      Retrieves statistics for a pair of nodes, including counts, functional connectivity, and distances.

      :param node1: The name of the first node.
      :type node1: str
      :param node2: The name of the second node.
      :type node2: str

      :returns: A tuple containing lists of counts, functional connectivity values,
                and distances between the two nodes.
      :rtype: tuple



   .. py:method:: _init_relay_distances()

      Calculates and stores the average distance between each node
      and the relay station, if a relay station is specified.



   .. py:method:: _init_nodes()

      Initializes and processes nodes using :py:meth:`src.config.nodes_processor.NodesProcessor.get_nodes_indices`.



   .. py:method:: _init_connectivity()

      Initializes the connectivity matrix and distances between nodes.

      It creates matrices for distances and connectivity weights
      between nodes, initializing with zeros or tuples as appropriate
      (depending on whether there is a relay station) and NaNs on the diagonal.



   .. py:method:: _load_data()

      Loads precomputed connectivity data such as counts, functional connectivity,
      and streamline lengths between nodes from the connectivity data path
      (see :py:class:`src.utils.paths.Paths`).



   .. py:method:: _init_data()

      Checks if the necessary structural and functional connectivity data files exist.
      If the files are found, it loads them; otherwise, it triggers the data preparation
      process using :py:class:`src.data_prep.data_preparator.DataPreparator`
      and then loads the prepared data.



.. py:class:: ModelConfig(nodes: list[str], relay_station: Optional[str], sample_rate: int, t_lags: int, t_secs: int, t_burnit: int, noise_color: str, std_noise: int, dist_shape: float, dist_scale: float, dist_location: float, dist_trunc_percent: float, custom_connectivity=False)

   A class to configure parameters and model the connectivity between brain nodes, including
   the distances, connectivity weights, and the generation of delays.

   .. attribute:: nodes

      The list of processed nodes used in the model.

      :type: list[str]

   .. attribute:: nr_nodes

      The total number of nodes in the model.

      :type: int

   .. attribute:: relay_station

      The relay station node name, if any.

      :type: str, optional

   .. attribute:: sample_rate

      The sampling rate of the model, in Hz.

      :type: int

   .. attribute:: nr_lags

      The number of time lags calculated based on the sample rate and the total time (ms) in lags.

      :type: int

   .. attribute:: t_secs

      The total time of the simulation in seconds.

      :type: int

   .. attribute:: t_burnit

      The burn-in time for the simulation, in seconds.

      :type: int

   .. attribute:: noise_color

      The color of the noise to be used in the simulation.

      :type: str

   .. attribute:: std_noise

      The standard deviation of the noise to be used in the simulation.

      :type: int

   .. attribute:: distances

      A matrix containing the distances between the nodes.

      :type: numpy.ndarray

   .. attribute:: connectivity_weights

      A matrix containing the connectivity weights between the nodes.

      :type: numpy.ndarray

   .. attribute:: delay_calculator

      An instance of the :py:class:`src.simulation.delay_calculator.DelayCalculator` class used to calculate delay distributions.

      :type: DelayCalculator

   .. attribute:: _dist_shape

      The shape parameter for the delay distribution (xi in GEV distribution).

      :type: float

   .. attribute:: _dist_scale

      The scale parameter for the delay distribution (sigma in GEV distribution).

      :type: float

   .. attribute:: _dist_location

      The location parameter for the delay distribution (mu in GEV distribution).

      :type: float

   .. attribute:: _truncation_percentile

      The percentile at which to truncate the delay distribution.

      :type: float


   .. py:method:: __init__(nodes: list[str], relay_station: Optional[str], sample_rate: int, t_lags: int, t_secs: int, t_burnit: int, noise_color: str, std_noise: int, dist_shape: float, dist_scale: float, dist_location: float, dist_trunc_percent: float, custom_connectivity=False)

      Initializes the ModelConfig with specified parameters for nodes, connectivity,
      simulation, and delay distribution.

      :param nodes: The list of nodes to be used in the connectivity model.
      :type nodes: list[str]
      :param relay_station: The relay station name, if any.
      :type relay_station: str, optional
      :param sample_rate: The sampling rate of the model, in Hz.
      :type sample_rate: int
      :param t_lags: The total time in lags for the simulation.
      :type t_lags: int
      :param t_secs: The total time of the simulation in seconds.
      :type t_secs: int
      :param t_burnit: The burn-in time for the simulation, in seconds.
      :type t_burnit: int
      :param noise_color: The color of the noise to be used in the simulation.
      :type noise_color: str
      :param std_noise: The standard deviation of the noise to be used in the simulation.
      :type std_noise: int
      :param dist_shape: The shape parameter for the delay distribution (xi in GEV distribution).
      :type dist_shape: float
      :param dist_scale: The scale parameter for the delay distribution (sigma in GEV distribution).
      :type dist_scale: float
      :param dist_location: The location parameter for the delay distribution (mu in GEV distribution).
      :type dist_location: float
      :param dist_trunc_percent: The percentile at which to truncate the delay distribution.
      :type dist_trunc_percent: float
      :param custom_connectivity: If True, use custom connectivity weights from a pre-specified file.
      :type custom_connectivity: bool, optional



   .. py:method:: __str__()

      Returns a string representation of the ModelConfig object, including details
      about the nodes, connectivity, simulation parameters, and GEV distribution parameters.

      :returns: A formatted string representation of the ModelConfig object.
      :rtype: str



   .. py:method:: plot(plots_dir)

      Plots (using :py:meth:`_plot_properties`) the connectivity model's distances
      (summed through the relay, if applicable) and normalized weights matrices using heatmaps.

      :param plots_dir: The directory where the plots are saved.
      :type plots_dir: pathlib.Path

      :raises AssertionError: If the plots directory does not exist.



   .. py:method:: _plot_properties(matrix, title, plots_dir, factor=1.0)

      Helper method to plot a heatmap of a given matrix with specified properties.

      :param matrix: The matrix to be plotted as a heatmap.
      :type matrix: numpy.ndarray
      :param title: The title for the plot, used to label the saved file.
      :type title: str
      :param plots_dir: The directory where the plots are saved.
      :type plots_dir: pathlib.Path
      :param factor: A scaling factor applied to the matrix values (default is 1.0).
      :type factor: float, optional



.. py:class:: NodesProcessor(given_nodes, relay_station)

   A class to process given brain regions into their corresponding indices in
   the Julich brain parcellation by finding the final generation children of
   the given nodes.

   .. attribute:: given_nodes

      List of nodes to be processed.

      :type: list

   .. attribute:: relay_station

      The relay station node, if any, that connects different brain regions.

      :type: str

   .. attribute:: _areas_dict

      A dictionary mapping brain areas to their hierarchical structure
      using :py:meth:`src.data_prep.areas_terminology_parser.AreasTerminologyParser.parse_into_dict`.

      :type: dict


   .. py:method:: __init__(given_nodes, relay_station)

      Initializes the NodesProcessor with the provided nodes, relay station, and whether to separate
      left and right brain nodes.

      :param given_nodes: List of nodes to be processed.
      :type given_nodes: list
      :param relay_station: The relay station node, if any.
      :type relay_station: str



   .. py:method:: get_nodes_indices()

      Processes the nodes and the relay station, if any, and retrieves their
      corresponding indices using :py:meth:`_process_nodes`.

      :returns: A tuple containing relay nodes, relay nodes indices, interaction nodes, and interaction nodes indices.
      :rtype: tuple



   .. py:method:: _process_nodes(nodes)

      Processes a list of nodes and retrieves their corresponding indices
      using :py:meth:`_get_nodes_and_indices`.

      :param nodes: List of nodes to be processed.
      :type nodes: list

      :returns: A tuple containing the processed nodes and their indices.
      :rtype: tuple



   .. py:method:: _get_nodes_and_indices(node)

      Retrieves the final generation nodes and their corresponding indices for a given brain region (node).
      Uses :py:meth:`_initialize_nodes_and_indices` to set up initial structures,
      :py:meth:`_find_final_generation_children` to locate the final generation nodes,
      and :py:meth:`_populate_nodes_indices` to assign indices based on the Julich brain parcellation.

      :param node: The name of the brain region (node) to retrieve indices for.
      :type node: str

      :returns: A tuple containing:

                - nodes: list of nodes corresponding to the brain region.
                - nodes_indices: dictionary mapping each node to its corresponding indices.
      :rtype: tuple



   .. py:method:: _initialize_nodes_and_indices(node)

      Initializes the nodes and their corresponding indices, considering left and right brain separation.

      :param node: The node to initialize.
      :type node: str

      :returns: A tuple containing the initialized nodes and their indices.
      :rtype: tuple



   .. py:method:: _find_final_generation_children(dictionary, target, found=False)

      Recursively finds the final generation children of a target node in the areas dictionary.

      :param dictionary: The dictionary containing hierarchical brain area mappings (initially :py:attr:`_areas_dict`).
      :type dictionary: dict
      :param target: The target node to find children for.
      :type target: str
      :param found: Whether the target node has been found (default is False).
      :type found: bool, optional

      :returns: A list of final generation children nodes.
      :rtype: list



   .. py:method:: _populate_nodes_indices(nodes_indices, children, node)

      Populates the nodes indices with the corresponding Julich labels.

      :param nodes_indices: The dictionary to populate with node indices.
      :type nodes_indices: dict
      :param children: The list of child nodes to process.
      :type children: list
      :param node: The original node being processed.
      :type node: str

      :returns: The populated nodes indices.
      :rtype: dict



.. py:class:: Parser(parser)

   The Parser class is responsible for parsing the command line arguments and setting
   default values.

   .. attribute:: parser

      The argument parser instance used to parse command line arguments.

      :type: argparse.ArgumentParser

   .. attribute:: _default_model_name

      The default name of the model.

      :type: str

   .. attribute:: _default_nodes

      The default brain areas where the nodes are placed.

      :type: str

   .. attribute:: _default_relay_station

      The default brain area to use as a relay station.

      :type: str

   .. attribute:: _default_custom_connectivity

      Indicates whether to use custom connectivity by default.

      :type: bool

   .. attribute:: _default_t_lags

      The default lagged time in milliseconds.

      :type: int

   .. attribute:: _default_sample_rate

      The default sample rate in Hz.

      :type: int

   .. attribute:: _default_t_secs

      The default simulation time in seconds.

      :type: int

   .. attribute:: _default_t_burnit

      The default number of seconds (burn-in) to delete for model convergence.

      :type: int

   .. attribute:: _default_noise_color

      The default color of the noise.

      :type: str

   .. attribute:: _default_std_noise

      The default standard deviation of the noise.

      :type: int

   .. attribute:: _default_dist_shape

      The default shape parameter for the lag distributions.

      :type: float

   .. attribute:: _default_dist_scale

      The default scale parameter for the lag distributions.

      :type: float

   .. attribute:: _default_dist_location

      The default location parameter for the lag distributions.

      :type: float

   .. attribute:: _default_dist_trunc_percent

      The default truncation percentile for the lag distributions.

      :type: float


   .. py:method:: __init__(parser)

      Initializes the Parser with an argparse parser and loads the default
      values from `model_params.yml` configuration file in the configs data path
      (see :py:class:`src.utils.paths.Paths`)

      :param parser: The argument parser instance used to parse command line arguments.
      :type parser: argparse.ArgumentParser



   .. py:method:: _load_yaml(file_path)
      :staticmethod:


      Loads a YAML file from the specified path.

      :param file_path: The path to the YAML file to load.
      :type file_path: str or pathlib.Path

      :returns: The contents of the YAML file as a dictionary.
      :rtype: dict



   .. py:method:: parse_args()

      Parses known arguments from the command line, validates them, and returns them
      as an argparse Namespace object.

      :returns: The parsed command line arguments.
      :rtype: argparse.Namespace



   .. py:method:: _add_arguments()

      Adds the command line arguments to the parser and sets their default values
      based on the loaded YAML configuration.



   .. py:method:: _validate_args(args)
      :staticmethod:


      Validates the parsed command line arguments.

      :param args: The parsed command line arguments.
      :type args: argparse.Namespace

      :raises ValueError: If the parsed arguments are invalid.



   .. py:method:: _parse_relay_station(relay_station_value: str)
      :staticmethod:


      Parses the relay station argument from the command line.
      If the provided value is "none", it returns None; otherwise, it returns the string value.

      :param relay_station_value: The relay station value provided from the command line.
      :type relay_station_value: str

      :returns: The parsed relay station value, or None if "none" is provided.
      :rtype: str or None



   .. py:method:: _parse_nodes(nodes_str)
      :staticmethod:


      Parses a string into a list of nodes.

      This method converts a semicolon-separated string of node names into a list of strings.
      For example: "node1; node2; node3" -> ["node1", "node2", "node3"].

      :param nodes_str: The semicolon-separated string of node names.
      :type nodes_str: str

      :returns: A list of node names.
      :rtype: list

      :raises argparse.ArgumentTypeError: If the input string cannot be parsed into a valid list of nodes.



