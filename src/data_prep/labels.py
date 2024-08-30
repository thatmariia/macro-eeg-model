# local imports
from utils.paths import paths


def populate_labels_julich():
    """
    Reads the raw labels from `labels_raw.txt` located in the Julich data path
    (see :py:class:`src.utils.paths.Paths`) and populates a dictionary with
    the labels as keys and their corresponding indices (adjusted by -1) as values.

    Returns
    -------
    dict
        A dictionary where the keys are labels (as strings) and the values are the corresponding
        indices (integers) adjusted by -1.
    """

    lj = {}
    # the index-1 as value and the rest of the words as key
    with open(paths.julich_data_path / "labels_raw.txt", "r") as file:
        for line in file:
            index, *label = line.split()
            lj[" ".join(label)] = int(index) - 1

    return lj


labels_julich = populate_labels_julich()
