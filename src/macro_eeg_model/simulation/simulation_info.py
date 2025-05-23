# external imports
import numpy as np


class SimulationInfo:
    """
    A class responsible for storing and retrieving information about a simulation.
    It handles saving and loading the data related to a simulation, such as nodes, distances, connectivity weights, and results.

    Attributes
    ----------
    nodes : numpy.ndarray
        The array of nodes used in the simulation.
    distances : numpy.ndarray
        The distance matrix between nodes used in the simulation.
    connectivity_weights : numpy.ndarray
        The connectivity weights matrix between nodes.
    sample_rate : int
        The sample rate of the simulation in Hz.
    lag_connectivity_weights : numpy.ndarray
        The lagged connectivity weights matrix used in the VAR model.
    simulation_data : numpy.ndarray
        The simulated EEG data.
    frequencies : numpy.ndarray
        The array of frequencies corresponding to the power spectrum.
    power : numpy.ndarray
        The power spectrum calculated from the simulation data.
    _output_dir : pathlib.Path
        The directory path where simulation results are saved.
    """

    def __init__(
            self,
            output_dir,
            nodes=None,
            distances=None,
            connectivity_weights=None,
            sample_rate=None,
            lag_connectivity_weights=None,
            simulation_data=None,
            frequencies=None,
            power=None
    ):
        """
        Initializes the SimulationInfo class with the provided simulation parameters and data.

        Parameters
        ----------
        output_dir: pathlib.Path
            The path to the output directory where simulation results are saved.
        nodes : numpy.ndarray, optional
            The array of nodes used in the simulation.
        distances : numpy.ndarray, optional
            The distance matrix between nodes used in the simulation.
        connectivity_weights : numpy.ndarray, optional
            The connectivity weights matrix between nodes.
        sample_rate : int, optional
            The sample rate of the simulation in Hz.
        lag_connectivity_weights : numpy.ndarray, optional
            The lagged connectivity weights matrix used in the VAR model.
        simulation_data : numpy.ndarray, optional
            The simulated EEG data.
        frequencies : numpy.ndarray, optional
            The array of frequencies corresponding to the power spectrum.
        power : numpy.ndarray, optional
            The power spectrum calculated from the simulation data.

        Raises
        ------
        AssertionError
            If the output directory does not exist.
        """

        assert output_dir.exists(), f"Directory not found: {output_dir}"

        self._output_dir = output_dir

        self.nodes = nodes
        self.distances = distances
        self.connectivity_weights = connectivity_weights
        self.sample_rate = sample_rate
        self.lag_connectivity_weights = lag_connectivity_weights
        self.simulation_data = simulation_data
        self.frequencies = frequencies
        self.power = power

    def save_simulation_info(self):
        """
        Saves the simulation data to the output directory as .npy files.
        The data includes nodes, distances, connectivity weights,
        sample rate, lag connectivity weights, simulation data,
        frequencies, and power spectrum.
        """

        np.save(self._output_dir / "nodes.npy", self.nodes)
        np.save(self._output_dir / "distances.npy", self.distances)
        np.save(self._output_dir / "connectivity_weights.npy", self.connectivity_weights)
        np.save(self._output_dir / "sample_rate.npy", self.sample_rate)
        np.save(self._output_dir / "lag_connectivity_weights.npy", self.lag_connectivity_weights)
        np.save(self._output_dir / "simulation_data.npy", self.simulation_data)
        np.save(self._output_dir / "frequencies.npy", self.frequencies)
        np.save(self._output_dir / "power.npy", self.power)

    def load_simulation_info(self):
        """
        Loads all the relevant data of the simulation from the output directory
        and assigns them to the corresponding attributes of the class.

        Raises
        ------
        FileNotFoundError
            If any of the required files are not found in the output directory.
        """

        try:
            self.nodes = np.load(self._output_dir / "nodes.npy", allow_pickle=True)
            self.distances = np.load(self._output_dir / "distances.npy", allow_pickle=True)
            self.connectivity_weights = np.load(self._output_dir / "connectivity_weights.npy", allow_pickle=True)
            self.sample_rate = int(np.load(self._output_dir / "sample_rate.npy", allow_pickle=True))
            self.lag_connectivity_weights = np.load(self._output_dir / "lag_connectivity_weights.npy", allow_pickle=True)
            self.simulation_data = np.load(self._output_dir / "simulation_data.npy", allow_pickle=True)
            self.frequencies = np.load(self._output_dir / "frequencies.npy", allow_pickle=True)
            self.power = np.load(self._output_dir / "power.npy", allow_pickle=True)
        except FileNotFoundError:
            raise FileNotFoundError(f"Simulation data not found in {self._output_dir}")
