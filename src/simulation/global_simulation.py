# local imports
from simulation.simulation_info import SimulationInfo
from simulation.stationary_model_developer import StationaryModelDeveloper
from simulation.simulator import Simulator
from simulation.data_processor import DataProcessor
from simulation.eeg_analyzer import EEGAnalyzer
from utils.paths import paths


class GlobalSimulation:
    """
    A class responsible for orchestrating the entire simulation process.
    It integrates the development of a stationary model, the simulation of EEG data, data processing, and data analysis.

    The class uses:
    - :py:class:`src.simulation.stationary_model_developer.StationaryModelDeveloper` to create a stationary model from the provided configuration.
    - :py:class:`src.simulation.simulator.Simulator` to generate synthetic EEG data based on the model.
    - :py:class:`src.simulation.data_processor.DataProcessor` to filter and segment the simulated data.
    - :py:class:`src.simulation.eeg_analyzer.EEGAnalyzer` to calculate and plot the power spectrum of the EEG data.
    - :py:class:`src.simulation.simulation_info.SimulationInfo` to save the simulation results.
    """

    def __init__(self, config):
        """
        Initializes the GlobalSimulation class with the provided configuration.

        Parameters
        ----------
        config : ModelConfig
            The configuration object containing parameters for the simulation
            (instance of the :py:class:`src.config.model_config.ModelConfig` class).
        """

        self.config = config

    def run(self, save_data=False, make_plots=False, verbose=False, simulation_name=None):
        """
        Runs the global simulation process, including model development, data simulation, processing, analysis, and optional saving/plotting.

        Parameters
        ----------
        save_data : bool, optional
            If True, saves the simulation results (default is False).
        make_plots : bool, optional
            If True, generates and saves plots of the connectivity and power spectrum (default is False).
        verbose : bool, optional
            If True, displays progress bars and detailed information during the simulation (default is False).
        simulation_name : str, optional
            The name of the simulation, used for saving the results (default is None).

        Returns
        -------
        tuple
            A tuple containing:
            - simulation_data (numpy.ndarray): The simulated EEG data.
            - frequencies (numpy.ndarray): The array of frequencies corresponding to the power spectrum.
            - power (numpy.ndarray): The power spectrum of the simulated EEG data.
        """

        if simulation_name is not None:
            output_dir = paths.output_path / simulation_name
            output_dir.mkdir(parents=True, exist_ok=True)

            plots_dir = paths.plots_path / simulation_name
            plots_dir.mkdir(parents=True, exist_ok=True)
        else:
            output_dir = paths.output_path
            plots_dir = paths.plots_path

        if make_plots:
            self.config.plot(plots_dir=plots_dir)

        # Developing a stationary model
        stationary_model_developer = StationaryModelDeveloper(
            nr_lags=self.config.nr_lags,
            nr_nodes=self.config.nr_nodes,
            nodes=self.config.nodes,
            distances=self.config.distances,
            connectivity_weights=self.config.connectivity_weights,
            sample_rate=self.config.sample_rate,
            delay_calculator=self.config.delay_calculator,
        )
        lag_connectivity_weights = stationary_model_developer.develop(
            verbose=verbose
        )

        if make_plots:
            stationary_model_developer.plot_connectivity(
                lag_connectivity_weights=lag_connectivity_weights,
                plots_dir=plots_dir
            )

        # Simulating the model
        simulator = Simulator(
            lag_connectivity_weights=lag_connectivity_weights,
            sample_rate=self.config.sample_rate,
            nr_lags=self.config.nr_lags,
            nr_nodes=self.config.nr_nodes,
            t_secs=self.config.t_secs,
            t_burnit=self.config.t_burnit,
            noise_color=self.config.noise_color,
            std_noise=self.config.std_noise
        )
        simulation_data = simulator.simulate(verbose)

        # Processing the data
        data_processor = DataProcessor()
        simulation_data = data_processor.filter_data(
            data=simulation_data,
            sample_rate=self.config.sample_rate,
            pass_frequency=1,
            stop_frequency=0.25
        )
        simulation_data_segmented = data_processor.segment_data(
            data=simulation_data,
            sample_rate=self.config.sample_rate,
            nr_nodes=self.config.nr_nodes
        )

        # Analyzing and plotting the data
        eeg_analyzer = EEGAnalyzer()
        frequencies, power = eeg_analyzer.calculate_power(simulation_data_segmented, sample_rate=self.config.sample_rate)

        if make_plots:
            eeg_analyzer.plot_power(
                frequencies=frequencies, power=power, nodes=self.config.nodes, plots_dir=plots_dir
            )

            print(f"The plots have been saved in the directory: {plots_dir}")

        if save_data:
            simulation_info = SimulationInfo(
                output_dir=output_dir,
                nodes=self.config.nodes,
                distances=self.config.distances,
                connectivity_weights=self.config.connectivity_weights,
                sample_rate=self.config.sample_rate,
                lag_connectivity_weights=lag_connectivity_weights,
                simulation_data=simulation_data,
                frequencies=frequencies,
                power=power
            )
            simulation_info.save_simulation_info()

            print(f"The simulation results have been saved in the directory: {output_dir}")

        return simulation_data, frequencies, power
