# external imports
import numpy as np

# local imports
from config.nodes_processor import NodesProcessor
from utils.paths import paths
from data_prep.data_preparator import DataPreparator


class ConnectivityModel:
    """
    A class to model the connectivity between brain nodes. It computes
    distances and connectivity weights between nodes, with optional
    relay stations.

    Attributes
    ----------
    nodes : list
        The processed list of nodes used in the model.
    nr_nodes : int
        The total number of nodes in the model.
    distances : numpy.ndarray
        The matrix of distances between nodes.
    connectivity_weights : numpy.ndarray
        The matrix of connectivity weights between nodes.
    _given_nodes : list
        The list of nodes provided for the connectivity model.
    _relay_station : str
        The relay station node name, if any.
    _relay_nodes : list
        The list of relay nodes derived from the relay station, if applicable.
    _relay_indices : list
        The indices of the relay nodes in the connectivity model.
    _nodes_indices : dict
        A dictionary mapping each node to its corresponding indices.
    _avg_counts : numpy.ndarray
        The average counts of connections between nodes.
    _avg_fc : numpy.ndarray
        The average functional connectivity between nodes.
    _avg_lengths : numpy.ndarray
        The average distances (lengths) between nodes.
    _relay_distances : dict
        The dictionary of average distances between nodes and the relay station.
    """

    def __init__(self, given_nodes, relay_station):
        """
        Initializes the ConnectivityModel with given nodes and an optional relay station.

        Parameters
        ----------
        given_nodes : list
            The list of nodes to be used in the connectivity model.
        relay_station : str
            The relay station name (or None).
        """

        self._given_nodes = given_nodes
        self._relay_station = relay_station

        self._relay_nodes = None
        self._relay_indices = None
        self.nodes = None
        self._nodes_indices = None
        self.nr_nodes = None

        self._avg_counts = None
        self._avg_fc = None
        self._avg_lengths = None

        self._relay_distances = None
        self.distances = None
        self.connectivity_weights = None

        self._init_data()
        self._init_nodes()
        self._init_connectivity()
        self._init_relay_distances()

    def set_connectivity(self, custom_connectivity):
        """
        Assigns streamline lengths to the distances matrix (relayed, if applicable) and
        weights to the connectivity matrix based on either
        custom-provided values or default calculations based on functional connectivity (FC).

        The values for a pair of nodes are extracted from :py:meth:`_get_pair_stats`.

        Parameters
        ----------
        custom_connectivity : bool
            If True, attempts to load and use custom connectivity weights from
            `connectivity_weights.csv` file in the configs path (see :py:class:`src.utils.paths.Paths`).

        Raises
        ------
        AssertionError
            If the shape of the custom connectivity matrix is incorrect or the matrix has been incorrectly constructed.
        """

        custom_connectivity_path = paths.configs_path / "connectivity_weights.csv"
        custom_connectivity_weights = None
        if custom_connectivity:
            try:
                custom_connectivity_weights = np.loadtxt(custom_connectivity_path, delimiter=",")

                assert custom_connectivity_weights.shape == (self.nr_nodes, self.nr_nodes), "Custom connectivity matrix has wrong shape."

                if not np.allclose(custom_connectivity_weights, custom_connectivity_weights.T):
                    # make the lower triangle equal to the upper triangle
                    custom_connectivity_weights = np.triu(custom_connectivity_weights) + np.triu(
                        custom_connectivity_weights, 1).T

            except:
                print(f"Could not load custom connectivity weights from {custom_connectivity_path}. Using FC connectivity weights.")
                custom_connectivity = False

        # loop through every pair of nodes to fill the symmetrical matrix
        for i in range(self.nr_nodes):
            for j in range(i + 1, self.nr_nodes):
                _, fcs, distances = self._get_pair_stats(self.nodes[i], self.nodes[j])

                if self._relay_station is not None:
                    distances_k, distances_l = zip(*distances)
                    distance = (np.mean(distances_k), np.mean(distances_l))
                else:
                    distance = np.mean(distances)

                self.distances[i, j] = self.distances[j, i] = distance

                if custom_connectivity:
                    connectivity_weight = custom_connectivity_weights[i, j] #* (0.9 ** 9)
                else:
                    pw = 1.5
                    connectivity_weight = (np.mean(fcs) ** pw) * (10 ** (pw - int(pw / 2))) * (0.9 ** 20)

                self.connectivity_weights[i, j] = self.connectivity_weights[j, i] = connectivity_weight

        assert np.isnan(self.connectivity_weights).sum() == self.nr_nodes, \
            f"Expected connectivity weights to have {self.nr_nodes} NaNs, but got {np.isnan(self.connectivity_weights).sum()}"

    def _get_pair_stats(self, node1, node2):
        """
        Retrieves statistics for a pair of nodes, including counts, functional connectivity, and distances.

        Parameters
        ----------
        node1 : str
            The name of the first node.
        node2 : str
            The name of the second node.

        Returns
        -------
        tuple
            A tuple containing lists of counts, functional connectivity values,
            and distances between the two nodes.
        """

        indices_list1 = self._nodes_indices[node1]
        indices_list2 = self._nodes_indices[node2]

        counts, fcs, distances = [], [], []

        for i in indices_list1:
            for j in indices_list2:

                if self._relay_station is None:
                    avg_dist = self._avg_lengths[i, j]
                else:
                    avg_dist = (self._relay_distances[i], self._relay_distances[j])

                counts.append(self._avg_counts[i, j])
                fcs.append(self._avg_fc[i, j])
                distances.append(avg_dist)

        return counts, fcs, distances

    def _init_relay_distances(self):
        """
        Calculates and stores the average distance between each node
        and the relay station, if a relay station is specified.
        """

        if self._relay_station is not None:

            # get all indices from self._nodes_indices
            all_indices = set(index for indices in self._nodes_indices.values() for index in indices)

            self._relay_distances = {}
            for i in all_indices:
                distances_node_relay = []
                for j in self._relay_indices:
                    distances_node_relay.append(self._avg_lengths[i, j])
                self._relay_distances[i] = np.mean(distances_node_relay)

    def _init_nodes(self):
        """
        Initializes and processes nodes using :py:meth:`src.config.nodes_processor.NodesProcessor.get_nodes_indices`.
        """

        nodes_processor = NodesProcessor(
            given_nodes=self._given_nodes,
            relay_station=self._relay_station
        )
        self._relay_nodes, self._relay_indices, self.nodes, self._nodes_indices = (
            nodes_processor.get_nodes_indices()
        )
        self.nr_nodes = len(self.nodes)

    def _init_connectivity(self):
        """
        Initializes the connectivity matrix and distances between nodes.

        It creates matrices for distances and connectivity weights
        between nodes, initializing with zeros or tuples as appropriate
        (depending on whether there is a relay station) and NaNs on the diagonal.
        """

        if self._relay_station is None:
            # initially 0's with nan on the diagonal
            self.distances = np.zeros((self.nr_nodes, self.nr_nodes))
        else:
            # initially each element with tuple (0, 0), nan on the diagonal
            self.distances = np.empty((self.nr_nodes, self.nr_nodes), dtype=object)
            for i in range(self.nr_nodes):
                for j in range(self.nr_nodes):
                    self.distances[i, j] = (0, 0)

        np.fill_diagonal(self.distances, np.nan)

        # initially 0's with np.nan on the diagonal
        self.connectivity_weights = np.zeros((self.nr_nodes, self.nr_nodes))
        np.fill_diagonal(self.connectivity_weights, np.nan)

    def _load_data(self):
        """
        Loads precomputed connectivity data such as counts, functional connectivity,
        and streamline lengths between nodes from the connectivity data path
        (see :py:class:`src.utils.paths.Paths`).
        """

        self._avg_counts = np.load(paths.connectivity_data_path / "avg_counts.npy")
        self._avg_fc = np.load(paths.connectivity_data_path / "avg_fc.npy")
        self._avg_lengths = np.load(paths.connectivity_data_path / "avg_lengths.npy")

    def _init_data(self):
        """
        Checks if the necessary structural and functional connectivity data files exist.
        If the files are found, it loads them; otherwise, it triggers the data preparation
        process using :py:class:`src.data_prep.data_preparator.DataPreparator`
        and then loads the prepared data.
        """

        try:
            self._load_data()
        except:
            data_preparator = DataPreparator()
            directory_sc = "structural_connectivity_data"
            directory_fc = "functional_connectivity_data"

            print("Preparing Julich structural connectivity data...")
            data_preparator.prep_and_save(
                directory_name=directory_sc,
                included_word="Lengths",
                delimiter=",",
                name="lengths"
            )
            data_preparator.prep_and_save(
                directory_name=directory_sc,
                included_word="Counts",
                delimiter=",",
                name="counts"
            )
            print("Preparing Julich functional connectivity data...")
            data_preparator.prep_and_save(
                directory_name=directory_fc,
                included_word="concatenated",
                delimiter=" ",
                name="fc"
            )
            self._load_data()

        print("Loaded connectivity data...")
