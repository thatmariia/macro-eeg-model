# local imports
from data_prep.labels import labels_julich
from data_prep.areas_terminology_parser import AreasTerminologyParser


class NodesProcessor:
    """
    A class to process given brain regions into their corresponding indices in
    the Julich brain parcellation by finding the final generation children of
    the given nodes.

    Attributes
    ----------
    given_nodes : list
        List of nodes to be processed.
    relay_station : str
        The relay station node, if any, that connects different brain regions.
    _areas_dict : dict
        A dictionary mapping brain areas to their hierarchical structure
        using :py:meth:`src.data_prep.areas_terminology_parser.AreasTerminologyParser.parse_into_dict`.
    """

    def __init__(self, given_nodes, relay_station):
        """
        Initializes the NodesProcessor with the provided nodes, relay station, and whether to separate
        left and right brain nodes.

        Parameters
        ----------
        given_nodes : list
            List of nodes to be processed.
        relay_station : str
            The relay station node, if any.
        """

        self.given_nodes = given_nodes
        self.relay_station = relay_station

        areas_terminology_parser = AreasTerminologyParser()
        self._areas_dict = areas_terminology_parser.parse_into_dict()
        # print self._areas_dict dict neatly
        # print(json.dumps(self._areas_dict, indent=4, sort_keys=True))

    def get_nodes_indices(self):
        """
        Processes the nodes and the relay station, if any, and retrieves their
        corresponding indices using :py:meth:`_process_nodes`.

        Returns
        -------
        tuple
            A tuple containing relay nodes, relay nodes indices, interaction nodes, and interaction nodes indices.
        """

        interaction_nodes, interaction_nodes_indices = self._process_nodes(self.given_nodes)

        if self.relay_station is not None:
            relay_nodes, relay_nodes_indices_d = self._process_nodes([self.relay_station])
            relay_nodes_indices = relay_nodes_indices_d[self.relay_station]
        else:
            relay_nodes, relay_nodes_indices = [], []

        return relay_nodes, relay_nodes_indices, interaction_nodes, interaction_nodes_indices

    def _process_nodes(self, nodes):
        """
        Processes a list of nodes and retrieves their corresponding indices
        using :py:meth:`_get_nodes_and_indices`.

        Parameters
        ----------
        nodes : list
            List of nodes to be processed.

        Returns
        -------
        tuple
            A tuple containing the processed nodes and their indices.
        """

        all_nodes = []
        all_nodes_indices = {}

        for node in nodes:
            nodes_list, nodes_indices = self._get_nodes_and_indices(node)
            all_nodes += nodes_list
            all_nodes_indices.update(nodes_indices)

        return all_nodes, all_nodes_indices

    def _get_nodes_and_indices(self, node):
        """
        Retrieves the final generation nodes and their corresponding indices for a given brain region (node).
        Uses :py:meth:`_initialize_nodes_and_indices` to set up initial structures,
        :py:meth:`_find_final_generation_children` to locate the final generation nodes,
        and :py:meth:`_populate_nodes_indices` to assign indices based on the Julich brain parcellation.

        Parameters
        ----------
        node : str
            The name of the brain region (node) to retrieve indices for.

        Returns
        -------
        tuple
            A tuple containing:
            - nodes: list of nodes corresponding to the brain region.
            - nodes_indices: dictionary mapping each node to its corresponding indices.
        """

        nodes, nodes_indices = self._initialize_nodes_and_indices(node)

        final_generation_children = self._find_final_generation_children(self._areas_dict, node)
        if not final_generation_children:
            final_generation_children = [node]

        nodes_indices = self._populate_nodes_indices(nodes_indices, final_generation_children, node)

        assert len(nodes_indices[node]) > 0, f"{node} is not in the Julich brain parcellation"

        return nodes, nodes_indices

    def _initialize_nodes_and_indices(self, node):
        """
        Initializes the nodes and their corresponding indices, considering left and right brain separation.

        Parameters
        ----------
        node : str
            The node to initialize.

        Returns
        -------
        tuple
            A tuple containing the initialized nodes and their indices.
        """

        nodes = []
        nodes_indices = {}

        nodes.append(node)
        nodes_indices[node] = []

        return nodes, nodes_indices

    def _find_final_generation_children(self, dictionary, target, found=False):
        """
        Recursively finds the final generation children of a target node in the areas dictionary.

        Parameters
        ----------
        dictionary : dict
            The dictionary containing hierarchical brain area mappings (initially :py:attr:`_areas_dict`).
        target : str
            The target node to find children for.
        found : bool, optional
            Whether the target node has been found (default is False).

        Returns
        -------
        list
            A list of final generation children nodes.
        """

        final_generation_children = []

        for key, value in dictionary.items():
            if found or key == target:
                if value:
                    for child in value:
                        final_generation_children += self._find_final_generation_children(child, target, True)
                else:
                    return [key] if found else []

                if key == target:
                    found = True
            elif value:
                for child in value:
                    if isinstance(child, dict):
                        final_generation_children += self._find_final_generation_children(child, target, False)

        return final_generation_children

    def _populate_nodes_indices(self, nodes_indices, children, node):
        """
        Populates the nodes indices with the corresponding Julich labels.

        Parameters
        ----------
        nodes_indices : dict
            The dictionary to populate with node indices.
        children : list
            The list of child nodes to process.
        node : str
            The original node being processed.

        Returns
        -------
        dict
            The populated nodes indices.
        """

        for child_node in children:
            left_child, right_child = child_node + " left", child_node + " right"

            if left_child in labels_julich and right_child in labels_julich:
                nodes_indices[node].extend([labels_julich[left_child], labels_julich[right_child]])
            # Uncomment if error handling is needed
            # else:
            #     raise ValueError(f"Node {node} has no left or right child with name {child_node}")

        return nodes_indices
