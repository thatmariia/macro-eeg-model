# local imports
from config.configs import get_simulate_config
from simulation.global_simulation import GlobalSimulation


def simulate():
    """
    The main function to run the simulation process.

    It retrieves the simulation configuration using :py:func:`src.config.configs.get_simulate_config`,
    initializes the :py:class:`src.simulation.global_simulation.GlobalSimulation` class,
    and executes the simulation using :py:meth:`src.simulation.global_simulation.GlobalSimulation.run`.
    """

    config, model_name = get_simulate_config()
    print(config)

    global_simulation = GlobalSimulation(config)
    _, _, _ = global_simulation.run(
        save_data=True, make_plots=True, verbose=True, simulation_name=model_name
    )


if __name__ == "__main__":
    simulate()
