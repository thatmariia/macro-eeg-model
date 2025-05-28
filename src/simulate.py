# local imports
from macro_eeg_model import get_simulate_config
from macro_eeg_model import GlobalSimulation
import argparse


def simulate():
    """
    The main function to run the simulation process.

    It retrieves the simulation configuration using :py:func:`src.config.configs.get_simulate_config`,
    initializes the :py:class:`src.simulation.global_simulation.GlobalSimulation` class,
    and executes the simulation using :py:meth:`src.simulation.global_simulation.GlobalSimulation.run`.
    """

    config, model_name, n = get_simulate_config()
    print(config)

    def run_simulation(name, make_plots=True, verbose=True):
        """
        Runs a single simulation using the provided configuration.
        """
        global_simulation = GlobalSimulation(config)
        _, _, _ = global_simulation.run(
            save_data=True, make_plots=True, verbose=True, simulation_name=name
        )

    if n == 1:
        print(f"Running single simulation: {model_name}...")
        run_simulation(model_name)
    else:
        print(f"Running batch simulation: {model_name} with {n} iterations...\n")
        for i in range(n):
            print(f"Running simulation {i + 1}/{n}...")
            single_model_name = f"{model_name}_{i + 1}"
            run_simulation(single_model_name, make_plots=False, verbose=False)


if __name__ == "__main__":
    simulate()
