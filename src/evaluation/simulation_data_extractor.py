# external imports
import numpy as np

# local imports
from utils.paths import paths
from simulation.simulation_info import SimulationInfo
from simulation.data_processor import DataProcessor


class SimulationDataExtractor:
    """
    The SimulationDataExtractor class is responsible for extracting and processing simulation data.
    It organizes the data by nodes and simulations, allowing for easy access to both raw and processed data.

    Attributes
    ----------
    nodes : numpy.ndarray
        An array of node names used in the simulations.
    simulation_names : list
        A list of simulation names.
    sample_rates : dict
        A dictionary mapping simulation names to their corresponding sample rates.
    simulations_data_per_node : dict
        A dictionary organizing the processed simulation data by node.
    simulations_power_per_node : dict
        A dictionary organizing the processed power spectra by node.
    """

    def __init__(self):
        """
        Initializes the SimulationDataExtractor by loading and processing the simulation data
        using methods from this class.
        """

        self.nodes = np.array(["frontal lobe", "parietal lobe", "occiptal lobe", "temporal lobe", "thalamus"])

        simulations_info, self.sample_rates = self._get_simulations_info()
        self.simulation_names = list(simulations_info.keys())
        self.simulation_names.sort()
        processed_simulations_data = self._get_processed_simulations_data(simulations_info)
        self.simulations_data_per_node = self._get_simulations_data_per_node(processed_simulations_data)
        processed_simulations_power = self._get_processed_simulations_power(simulations_info)
        self.simulations_power_per_node = self._get_simulations_power_per_node(processed_simulations_power)

    def _get_simulations_data_per_node(self, processed_simulations_data):
        """
        Organizes the processed simulation data by node and then simulation name.

        Parameters
        ----------
        processed_simulations_data : dict
            The dictionary containing processed simulation data organized by simulation name and then node.

        Returns
        -------
        dict
            A dictionary organizing the simulation data by node and then simulation name.
        """

        simulations_data_per_node = {
            node: {
                simulation_name: processed_simulations_data[simulation_name][node] for simulation_name in
                processed_simulations_data.keys()
            }
            for node in self.nodes[:-1]
        }

        return simulations_data_per_node

    def _get_simulations_power_per_node(self, processed_simulations_power):
        """
        Organizes the processed power spectra by node and then simulation name.

        Parameters
        ----------
        processed_simulations_power : dict
            The dictionary containing processed power spectra organized by simulation name and then node.

        Returns
        -------
        dict
            A dictionary organizing the power spectra by node and then simulation name.
        """

        simulations_power_per_node = {
            node: {
                simulation_name: processed_simulations_power[simulation_name][node] for simulation_name in
                processed_simulations_power.keys()
            }
            for node in self.nodes[:-1]
        }

        return simulations_power_per_node

    @staticmethod
    def _get_processed_simulations_power(simulations_info):
        """
        Processes and organizes the power spectra data by simulation name and then node.

        Parameters
        ----------
        simulations_info : dict
            A dictionary containing simulation information objects.

        Returns
        -------
        dict
            A dictionary organizing the processed power spectra data by simulation name and then node.
        """

        processed_simulations_power = {}

        for simulation_name, simulation_info in simulations_info.items():
            simulations_frequencies = simulation_info.frequencies
            simulations_power = simulation_info.power

            # swap dimensions of simulations_power
            simulations_power = np.swapaxes(simulations_power, 0, 1)

            processed_simulations_power[simulation_name] = {
                node: (np.array(simulations_frequencies), np.array(simulations_power[i]))
                for i, node in enumerate(simulation_info.nodes[:-1])
            }

        return processed_simulations_power


    @staticmethod
    def _get_processed_simulations_data(simulations_info):
        """
        Processes and organizes the raw simulation data by simulation name and then node.

        Parameters
        ----------
        simulations_info : dict
            A dictionary containing simulation information objects.

        Returns
        -------
        dict
            A dictionary organizing the processed simulation data by simulation name and then node.
        """

        processed_simulations_data = {}

        for simulation_name, simulation_info in simulations_info.items():
            simulation_data = simulation_info.simulation_data
            sample_rate = simulation_info.sample_rate
            nr_nodes = len(simulation_info.nodes)

            data_processor = DataProcessor()
            data = data_processor.segment_data(simulation_data, sample_rate=sample_rate, nr_nodes=nr_nodes)

            # data has shape (nr samples, nr nodes, nr epochs)
            # remove last node (relay station)
            data = data[:, :-1, :]

            # reshape data to (nr nodes, nr epochs, nr samples)
            data = np.transpose(data, (1, 2, 0))

            # reshape to include x seconds (nr nodes, nr epochs * y, x = nr samples / y)
            nr_secs = 2
            data = np.reshape(data, (data.shape[0], data.shape[1] // nr_secs, nr_secs * sample_rate))

            processed_simulations_data[simulation_name] = {
                node: np.array(data[i]) for i, node in enumerate(simulation_info.nodes[:-1])
            }

        return processed_simulations_data

    def _get_simulations_info(self):
        """
        Loads simulation information from saved files in the directories within the
        output path (see :py:class:`src.utils.paths.Paths`) using
        :py:meth:`src.simulation.simulation_info.SimulationInfo.load_simulation_info`.
        and checks for consistency in node names.

        Returns
        -------
        tuple
            A tuple containing:
            - simulations_info (dict): A dictionary of SimulationInfo objects keyed by simulation name.
            - sample_rates (dict): A dictionary of sample rates keyed by simulation name.

        Raises
        ------
        AssertionError
            If the nodes in any simulation do not match the expected node names.
        """

        simulations_info = {}
        sample_rates = {}

        for folder in paths.output_path.iterdir():
            if folder.is_dir():
                output_simulation_dir = paths.output_path / folder.name
                simulation_info = SimulationInfo(output_dir=output_simulation_dir)
                simulation_info.load_simulation_info()

                assert all(simulation_info.nodes == self.nodes), \
                    f"Nodes do not match for simulation {folder.name}. Expected {self.nodes}, got {simulation_info.nodes}"

                simulations_info[folder.name] = simulation_info
                sample_rates[folder.name] = simulation_info.sample_rate

        return simulations_info, sample_rates
