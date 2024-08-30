# standard imports
import argparse

# local imports
from config.parser import Parser
from config.model_config import ModelConfig


def get_simulate_config():
    """
    First parses command line arguments using :py:func:`get_parsed_args`, then
    creates an instance of the :py:class:`src.config.model_config.ModelConfig`
    class based on the parsed arguments.

    Returns
    -------
    tuple
        A tuple containing:
        - config (ModelConfig): The configuration object for the simulation.
        - model_name (str): The name of the model as specified in the command line arguments.
    """

    args = get_parsed_args()
    config = ModelConfig(
        nodes=args.nodes,
        relay_station=args.relay_station,
        sample_rate=args.sample_rate,
        t_lags=args.t_lags,
        t_secs=args.t_secs,
        t_burnit=args.t_burnit,
        noise_color=args.noise_color,
        std_noise=args.std_noise,
        dist_shape=args.dist_shape,
        dist_scale=args.dist_scale,
        dist_location=args.dist_location,
        dist_trunc_percent=args.dist_trunc_percent,
        custom_connectivity=args.custom_connectivity
    )
    return config, args.model_name


def get_parsed_args():
    """
    Parses command line arguments using :py:meth:`src.config.parser.Parser.parse_args`.

    Returns
    -------
    argparse.Namespace
        The parsed command line arguments as an `argparse.Namespace` object.
    """

    parser = argparse.ArgumentParser(prog="macro-eeg-model")
    parser = Parser(parser)
    args = parser.parse_args()
    return args
