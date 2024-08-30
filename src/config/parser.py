# standard imports
import argparse
import ast

# external imports
import yaml

# local imports
from utils.paths import paths


class Parser:
    """
    The Parser class is responsible for parsing the command line arguments and setting
    default values.

    Attributes
    ----------
    parser : argparse.ArgumentParser
        The argument parser instance used to parse command line arguments.
    _default_model_name : str
        The default name of the model.
    _default_nodes : str
        The default brain areas where the nodes are placed.
    _default_relay_station : str
        The default brain area to use as a relay station.
    _default_custom_connectivity : bool
        Indicates whether to use custom connectivity by default.
    _default_t_lags : int
        The default lagged time in milliseconds.
    _default_sample_rate : int
        The default sample rate in Hz.
    _default_t_secs : int
        The default simulation time in seconds.
    _default_t_burnit : int
        The default number of seconds (burn-in) to delete for model convergence.
    _default_noise_color : str
        The default color of the noise.
    _default_std_noise : int
        The default standard deviation of the noise.
    _default_dist_shape : float
        The default shape parameter for the lag distributions.
    _default_dist_scale : float
        The default scale parameter for the lag distributions.
    _default_dist_location : float
        The default location parameter for the lag distributions.
    _default_dist_trunc_percent : float
        The default truncation percentile for the lag distributions.
    """

    def __init__(self, parser):
        """
        Initializes the Parser with an argparse parser and loads the default
        values from `model_params.yml` configuration file in the configs data path
        (see :py:class:`src.utils.paths.Paths`)

        Parameters
        ----------
        parser : argparse.ArgumentParser
            The argument parser instance used to parse command line arguments.
        """

        self.parser = parser

        provided_model_params = self._load_yaml(paths.configs_path / "model_params.yml")

        # model params
        self._default_model_name = provided_model_params.get("model_name", "Simulated macro EEG model")

        self._default_nodes = provided_model_params.get("nodes", "frontal lobe; parietal lobe; occiptal lobe; temporal lobe; thalamus")
        self._default_relay_station = provided_model_params.get("relay_station", "none")
        self._default_custom_connectivity = provided_model_params.get("custom_connectivity", True)

        self._default_t_lags = provided_model_params.get("t_lags", 300)
        self._default_sample_rate = provided_model_params.get("sample_rate", 1000)
        self._default_t_secs = provided_model_params.get("t_secs", 500)
        self._default_t_burnit = provided_model_params.get("t_burnit", 10)
        self._default_noise_color = provided_model_params.get("noise_color", "white")
        self._default_std_noise = provided_model_params.get("std_noise", 50)

        self._default_dist_shape = provided_model_params.get("dist_shape", -0.25)
        self._default_dist_scale = provided_model_params.get("dist_scale", 0.09)
        self._default_dist_location = provided_model_params.get("dist_location", 0.25)

        self._default_dist_trunc_percent = provided_model_params.get("dist_trunc_percent", 0.0)

        self._add_arguments()

    @staticmethod
    def _load_yaml(file_path):
        """
        Loads a YAML file from the specified path.

        Parameters
        ----------
        file_path : str or pathlib.Path
            The path to the YAML file to load.

        Returns
        -------
        dict
            The contents of the YAML file as a dictionary.
        """

        with open(file_path, "r") as file:
            return yaml.safe_load(file)

    def parse_args(self):
        """
        Parses known arguments from the command line, validates them, and returns them
        as an argparse Namespace object.

        Returns
        -------
        argparse.Namespace
            The parsed command line arguments.
        """

        # parse known arguments to check for values
        args, _ = self.parser.parse_known_args()
        self._validate_args(args)
        return args

    def _add_arguments(self):
        """
        Adds the command line arguments to the parser and sets their default values
        based on the loaded YAML configuration.
        """

        self.parser.add_argument(
            "--model_name",
            type=str,
            default=self._default_model_name,
            help="name of the model"
        )
        self.parser.add_argument(
            "--nodes",
            type=self._parse_nodes,
            default=self._default_nodes,
            help="brain areas where the nodes are placed (according to Julich brain labels)"
        )
        self.parser.add_argument(
            "--relay_station",
            type=self._parse_relay_station,
            default=self._default_relay_station,
            help="brain area to use as a relay station (according to Julich brain labels or 'none')"
        )
        self.parser.add_argument(
            "--custom_connectivity",
            type=bool,
            default=self._default_custom_connectivity,
            help="whether to use custom connectivity (from connectivity_weights.csv) or not"
        )
        self.parser.add_argument(
            "--sample_rate",
            type=int,
            default=self._default_sample_rate,
            help="sample rate"
        )
        self.parser.add_argument(
            "--t_lags",
            type=int,
            default=self._default_t_lags,
            help="lagged time in ms"
        )
        self.parser.add_argument(
            "--t_secs",
            type=int,
            default=self._default_t_secs,
            help="simulation time in seconds"
        )
        self.parser.add_argument(
            "--t_burnit",
            type=int,
            default=self._default_t_burnit,
            help="number of seconds to delete to ensure model convergence"
        )
        self.parser.add_argument(
            "--noise_color",
            type=str,
            default=self._default_noise_color,
            help="color of the noise ('white' or 'pink')"
        )
        self.parser.add_argument(
            "--std_noise",
            type=float,
            default=self._default_std_noise,
            help="scalar standard deviation of the noise (effectively controls the scale of the output)"
        )
        self.parser.add_argument(
            "--dist_shape",
            type=float,
            default=self._default_dist_shape,
            help="shape param for the lag distributions"
        )
        self.parser.add_argument(
            "--dist_scale",
            type=float,
            default=self._default_dist_scale,
            help="scale param for the lag distributions"
        )
        self.parser.add_argument(
            "--dist_location",
            type=float,
            default=self._default_dist_location,
            help="location param for the lag distributions"
        )
        self.parser.add_argument(
            "--dist_trunc_percent",
            type=float,
            default=self._default_dist_trunc_percent,
            help="tail truncation percentile for the lag distributions"
        )

    @staticmethod
    def _validate_args(args):
        """
        Validates the parsed command line arguments.

        Parameters
        ----------
        args : argparse.Namespace
            The parsed command line arguments.

        Raises
        ------
        ValueError
            If the parsed arguments are invalid.
        """

        if args.relay_station is not None and args.relay_station not in args.nodes:
            raise ValueError("The relay station must be one of the nodes")

        if args.dist_trunc_percent < 0.0 or args.dist_trunc_percent >= 1.0:
            raise ValueError("The truncation percentile must be between 0 and 1")

        if args.noise_color not in ["white", "pink"]:
            raise ValueError("Invalid noise color. Must be 'white' or 'pink'")

        if args.std_noise <= 0:
            raise ValueError("The standard deviation of the noise must be positive")

        if args.t_lags <= 0:
            raise ValueError("The lagged time must be positive")

        if args.t_secs <= 0:
            raise ValueError("The simulation time must be positive")

        if args.t_burnit <= 0:
            raise ValueError("The burn-in time must be positive")

        if args.sample_rate <= 0:
            raise ValueError("The sample rate must be positive")

        if not bool((args.dist_scale > 0) and (args.dist_shape != 0)):
            raise ValueError("Invalid distribution parameters. Scale must be positive and shape must not be zero")

    @staticmethod
    def _parse_relay_station(relay_station_value: str):
        """
        Parses the relay station argument from the command line.
        If the provided value is "none", it returns None; otherwise, it returns the string value.

        Parameters
        ----------
        relay_station_value : str
            The relay station value provided from the command line.

        Returns
        -------
        str or None
            The parsed relay station value, or None if "none" is provided.
        """

        if relay_station_value == "none":
            return None
        return relay_station_value

    @staticmethod
    def _parse_nodes(nodes_str):
        """
        Parses a string into a list of nodes.

        This method converts a semicolon-separated string of node names into a list of strings.
        For example: "node1; node2; node3" -> ["node1", "node2", "node3"].

        Parameters
        ----------
        nodes_str : str
            The semicolon-separated string of node names.

        Returns
        -------
        list
            A list of node names.

        Raises
        ------
        argparse.ArgumentTypeError
            If the input string cannot be parsed into a valid list of nodes.
        """

        # remove spaces after semicolons
        nodes_str = nodes_str.replace("; ", ";")

        # add quotes to each element and brackets to the beginning and end
        nodes_str = nodes_str.replace(";", "\",\"")
        nodes_str = "[\"" + nodes_str + "\"]"

        try:
            nodes = ast.literal_eval(nodes_str)
            return nodes
        except ValueError:
            raise argparse.ArgumentTypeError("Invalid nodes argument input format")
