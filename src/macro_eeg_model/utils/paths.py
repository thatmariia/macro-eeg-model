# standard imports
from pathlib import Path


class Paths:
    """
    A class responsible for managing directory paths used in the project.
    It ensures that the necessary directories exist, creating them if they do not.

    Attributes
    ----------
    base_dir : pathlib.Path
        The base directory for all project-related paths. Defaults to the current working directory.
    configs_path : pathlib.Path
        The path to the 'configs' directory, used for storing configuration files.
    connectivity_data_path : pathlib.Path
        The path to the 'connectivity_data' directory, used for storing connectivity-related data.
    julich_data_path : pathlib.Path
        The path to the 'julich_brain_data' directory, used for storing Julich brain data.
    plots_path : pathlib.Path
        The path to the 'plots' directory, used for storing generated plots.
    output_path : pathlib.Path
        The path to the 'output' directory, used for storing output files and results.
    """

    def __init__(self, base_dir=None):
        """
        Initializes the Paths object, setting up the base directory and subdirectories.

        Parameters
        ----------
        base_dir : str or pathlib.Path, optional
            The base directory for the project. If not provided, the current working directory is used.

        Raises
        ------
        AssertionError
            If the 'configs' or 'julich_brain_data' directories do not exist within the base directory.
        """

        self.base_dir = Path(base_dir) if base_dir else Path.cwd()

        self.configs_path = self.base_dir / "configs"
        self.connectivity_data_path = self.base_dir / "connectivity_data"
        self.julich_data_path = self.base_dir / "julich_brain_data"
        self.plots_path = self.base_dir / "plots"
        self.output_path = self.base_dir / "output"

        assert self.configs_path.exists(), f"Directory not found: {self.configs_path}"
        assert self.julich_data_path.exists(), f"Directory not found: {self.connectivity_data_path}"

        self.connectivity_data_path.mkdir(exist_ok=True)
        self.plots_path.mkdir(exist_ok=True)
        self.output_path.mkdir(exist_ok=True)


paths = Paths()
