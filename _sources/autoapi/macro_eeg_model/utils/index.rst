macro_eeg_model.utils
=====================

.. py:module:: macro_eeg_model.utils

.. autoapi-nested-parse::

   utils
   ------
   This package contains the utility functions (for plotting and path handling).



Submodules
----------

.. toctree::
   :maxdepth: 1

   /autoapi/macro_eeg_model/utils/paths/index
   /autoapi/macro_eeg_model/utils/plotting_setup/index


Attributes
----------

.. autoapisummary::

   macro_eeg_model.utils.paths
   macro_eeg_model.utils.PLOT_SIZE
   macro_eeg_model.utils.PLOT_FORMAT
   macro_eeg_model.utils.COLORS


Classes
-------

.. autoapisummary::

   macro_eeg_model.utils.Paths


Functions
---------

.. autoapisummary::

   macro_eeg_model.utils.notation


Package Contents
----------------

.. py:class:: Paths(base_dir=None)

   A class responsible for managing directory paths used in the project.
   It ensures that the necessary directories exist, creating them if they do not.

   .. attribute:: base_dir

      The base directory for all project-related paths. Defaults to the current working directory.

      :type: pathlib.Path

   .. attribute:: configs_path

      The path to the 'configs' directory, used for storing configuration files.

      :type: pathlib.Path

   .. attribute:: connectivity_data_path

      The path to the 'connectivity_data' directory, used for storing connectivity-related data.

      :type: pathlib.Path

   .. attribute:: julich_data_path

      The path to the 'julich_brain_data' directory, used for storing Julich brain data.

      :type: pathlib.Path

   .. attribute:: plots_path

      The path to the 'plots' directory, used for storing generated plots.

      :type: pathlib.Path

   .. attribute:: output_path

      The path to the 'output' directory, used for storing output files and results.

      :type: pathlib.Path


   .. py:method:: __init__(base_dir=None)

      Initializes the Paths object, setting up the base directory and subdirectories.

      :param base_dir: The base directory for the project. If not provided, the current working directory is used.
      :type base_dir: str or pathlib.Path, optional

      :raises AssertionError: If the 'configs' or 'julich_brain_data' directories do not exist within the base directory.



.. py:data:: paths

.. py:data:: PLOT_SIZE
   :value: 10


.. py:data:: PLOT_FORMAT
   :value: 'pdf'


.. py:data:: COLORS
   :value: ['#E64B35', '#00A087', '#3C5488', '#FFA70F', '#208BB5', '#ED7287', '#6AC882', '#FF7C2B',...


.. py:function:: notation(region_name)

   Create an abbreation for a region name.

   :param region_name: The name of the region.
   :type region_name: str


