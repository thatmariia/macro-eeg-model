# standard imports
import json

# local imports
from utils.paths import paths


class AreasTerminologyParser:
    """
    A class to parse the Julich hierarchical parcellation terminology into a dictionary.
    """

    @staticmethod
    def parse_into_dict():
        """
        Parses the `areas_terminology.json` file located in the Julich data path
        (see :py:class:`src.utils.paths.Paths`) into a nested dictionary.

        The method reads a JSON file containing the hierarchical structure of
        brain areas, processes the data, and returns it in a clean dictionary
        format.

        Returns
        -------
        dict
            A nested dictionary where each key represents a brain area and its
            corresponding children areas are stored as values.
        """

        json_path = paths.julich_data_path / "areas_terminology.json"
        with open(json_path, "r") as file:
            areas_dict = json.load(file)
        areas_dict["children"] = areas_dict["properties"]["regions"]

        def convert_to_dict(item):
            # Base case: if the item has no children, return its name
            if ('children' not in item) or (not item['children']):
                return {item['name']: []}

            # Recursive case: the item has children
            children_list = [convert_to_dict(child) for child in item['children']]
            return {item['name']: children_list}

        clean_areas_dict = convert_to_dict(areas_dict)

        return clean_areas_dict
