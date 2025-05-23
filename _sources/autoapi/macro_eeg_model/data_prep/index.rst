macro_eeg_model.data_prep
=========================

.. py:module:: macro_eeg_model.data_prep

.. autoapi-nested-parse::

   data_prep
   ---------
   This package contains the Julich brain data preparation scripts for the simulation.



Submodules
----------

.. toctree::
   :maxdepth: 1

   /autoapi/macro_eeg_model/data_prep/areas_terminology_parser/index
   /autoapi/macro_eeg_model/data_prep/data_preparator/index
   /autoapi/macro_eeg_model/data_prep/labels/index


Attributes
----------

.. autoapisummary::

   macro_eeg_model.data_prep.labels_julich


Classes
-------

.. autoapisummary::

   macro_eeg_model.data_prep.AreasTerminologyParser
   macro_eeg_model.data_prep.DataPreparator


Functions
---------

.. autoapisummary::

   macro_eeg_model.data_prep.populate_labels_julich


Package Contents
----------------

.. py:class:: AreasTerminologyParser

   A class to parse the Julich hierarchical parcellation terminology into a dictionary.


   .. py:method:: parse_into_dict()
      :staticmethod:


      Parses the `areas_terminology.json` file located in the Julich data path
      (see :py:class:`src.utils.paths.Paths`) into a nested dictionary.

      The method reads a JSON file containing the hierarchical structure of
      brain areas, processes the data, and returns it in a clean dictionary
      format.

      :returns: A nested dictionary where each key represents a brain area and its
                corresponding children areas are stored as values.
      :rtype: dict



.. py:class:: DataPreparator

   A class to prepare and process data from directories containing CSV files with
   connectivity data across subjects.
   The processed data is saved as a NumPy array after averaging across multiple subjects.


   .. py:method:: prep_and_save(directory_name, included_word, delimiter, name)

      Handles the prerequisites for preparing and saving the data from a specified directory
      within the Julich data path (see :py:class:`src.utils.paths.Paths`)
      and then does the actual data preparation and saving using :py:meth:`_prep_and_save_data`.

      This method filters the files in the directory based on an included word in their filenames,
      processes them into NumPy arrays, calculates an average array, and saves it to a specified path.

      :param directory_name: The name of the directory containing the subject folders.
      :type directory_name: str
      :param included_word: The word that should be included in the CSV filenames to be processed.
      :type included_word: str
      :param delimiter: The delimiter used in the CSV files.
      :type delimiter: str
      :param name: The name to use when saving the final averaged array.
      :type name: str



   .. py:method:: _prep_and_save_data(directory, subjects, included_word, delimiter, name)

      Extracts relevant CSV files based on the included word using :py:meth:`_extract_csv_files`
      converts them to NumPy arrays using :py:meth:`_get_arrays_from_files`,
      computes an average array using :py:meth:`_calculate_avg_array`,
      and saves it as a .npy file.

      :param directory: The path to the directory containing the subject folders.
      :type directory: str or pathlib.Path
      :param subjects: The list of subject folder names.
      :type subjects: list
      :param included_word: The word that should be included in the CSV filenames to be processed.
      :type included_word: str
      :param delimiter: The delimiter used in the CSV files.
      :type delimiter: str
      :param name: The name to use when saving the final averaged array.
      :type name: str



   .. py:method:: _extract_csv_files(directory, subjects, included_word)
      :staticmethod:


      Extracts the names of CSV files that include a specific word in their filenames.
      Searches through the directory of each subject for CSV files that contain
      the specified word in their name.

      :param directory: The path to the directory containing the subject folders.
      :type directory: str or pathlib.Path
      :param subjects: The list of subject folder names.
      :type subjects: list
      :param included_word: The word that must be included in the filenames.
      :type included_word: str

      :returns: A list of filenames that match the criteria.
      :rtype: list



   .. py:method:: _get_arrays_from_files(directory, subjects, files, delimiter=',')

      Retrieves and converts the relevant CSV files into NumPy arrays
      using :py:meth:`_convert_csv_file_to_numpy_array`.

      For each subject in the directory, this method identifies the files to be processed,
      converts them into NumPy arrays, and collects them for further processing.

      :param directory: The path to the directory containing the subject folders.
      :type directory: str or pathlib.Path
      :param subjects: The list of subject folder names.
      :type subjects: list
      :param files: The list of filenames to be processed.
      :type files: list
      :param delimiter: The delimiter used in the CSV files (default is ',').
      :type delimiter: str, optional

      :returns: A list of NumPy arrays corresponding to the processed CSV files.
      :rtype: list



   .. py:method:: _convert_csv_file_to_numpy_array(file_path, delimiter)
      :staticmethod:


      Converts a CSV file into a NumPy array.

      :param file_path: The full path to the CSV file.
      :type file_path: str or pathlib.Path
      :param delimiter: The delimiter used in the CSV file.
      :type delimiter: str

      :returns: A NumPy array representing the data from the CSV file.
      :rtype: numpy.ndarray



   .. py:method:: _calculate_avg_array(numpy_arrays)
      :staticmethod:


      Computes the average of each element across multiple NumPy arrays,
      excluding the highest and lowest 20% of values (to reduce the impact of outliers),
      and returns the resulting array.

      :param numpy_arrays: A list of NumPy arrays to average.
      :type numpy_arrays: list

      :returns: A NumPy array containing the average values.
      :rtype: numpy.ndarray



.. py:function:: populate_labels_julich()

   Reads the raw labels from `labels_raw.txt` located in the Julich data path
   (see :py:class:`src.utils.paths.Paths`) and populates a dictionary with
   the labels as keys and their corresponding indices (adjusted by -1) as values.

   :returns: A dictionary where the keys are labels (as strings) and the values are the corresponding
             indices (integers) adjusted by -1.
   :rtype: dict


.. py:data:: labels_julich

