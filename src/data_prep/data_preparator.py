# standard imports
import os

# external imports
import numpy as np

# local imports
from utils.paths import paths


class DataPreparator:
    """
    A class to prepare and process data from directories containing CSV files with
    connectivity data across subjects.
    The processed data is saved as a NumPy array after averaging across multiple subjects.
    """

    def prep_and_save(self, directory_name, included_word, delimiter, name):
        """
        Handles the prerequisites for preparing and saving the data from a specified directory
        within the Julich data path (see :py:class:`src.utils.paths.Paths`)
        and then does the actual data preparation and saving using :py:meth:`_prep_and_save_data`.

        This method filters the files in the directory based on an included word in their filenames,
        processes them into NumPy arrays, calculates an average array, and saves it to a specified path.

        Parameters
        ----------
        directory_name : str
            The name of the directory containing the subject folders.
        included_word : str
            The word that should be included in the CSV filenames to be processed.
        delimiter : str
            The delimiter used in the CSV files.
        name : str
            The name to use when saving the final averaged array.
        """

        directory = paths.julich_data_path / directory_name
        subjects = os.listdir(directory)

        def is_subject(subject):
            return (subject != ".DS_Store") and (subject != ".gitkeep")

        # discard ds store
        subjects = [subject for subject in subjects if is_subject(subject)]
        subjects.sort()

        self._prep_and_save_data(directory, subjects, included_word, delimiter, name)

    def _prep_and_save_data(self, directory, subjects, included_word, delimiter, name):
        """
        Extracts relevant CSV files based on the included word using :py:meth:`_extract_csv_files`
        converts them to NumPy arrays using :py:meth:`_get_arrays_from_files`,
        computes an average array using :py:meth:`_calculate_avg_array`,
        and saves it as a .npy file.

        Parameters
        ----------
        directory : str or pathlib.Path
            The path to the directory containing the subject folders.
        subjects : list
            The list of subject folder names.
        included_word : str
            The word that should be included in the CSV filenames to be processed.
        delimiter : str
            The delimiter used in the CSV files.
        name : str
            The name to use when saving the final averaged array.
        """

        csv_files = list(set(self._extract_csv_files(directory, subjects, included_word)))
        numpy_arrays = self._get_arrays_from_files(directory, subjects, csv_files, delimiter)
        avg_array = self._calculate_avg_array(numpy_arrays)
        path = paths.connectivity_data_path / f"avg_{name}.npy"
        np.save(path, avg_array)

    @staticmethod
    def _extract_csv_files(directory, subjects, included_word):
        """
        Extracts the names of CSV files that include a specific word in their filenames.
        Searches through the directory of each subject for CSV files that contain
        the specified word in their name.

        Parameters
        ----------
        directory : str or pathlib.Path
            The path to the directory containing the subject folders.
        subjects : list
            The list of subject folder names.
        included_word : str
            The word that must be included in the filenames.

        Returns
        -------
        list
            A list of filenames that match the criteria.
        """

        csv_files = []
        for subject in subjects:
            for filename in os.listdir(os.path.join(directory, subject)):
                if filename.endswith(".csv") and included_word in filename:
                    csv_files.append(filename)

        return csv_files

    def _get_arrays_from_files(self, directory, subjects, files, delimiter=","):
        """
        Retrieves and converts the relevant CSV files into NumPy arrays
        using :py:meth:`_convert_csv_file_to_numpy_array`.

        For each subject in the directory, this method identifies the files to be processed,
        converts them into NumPy arrays, and collects them for further processing.

        Parameters
        ----------
        directory : str or pathlib.Path
            The path to the directory containing the subject folders.
        subjects : list
            The list of subject folder names.
        files : list
            The list of filenames to be processed.
        delimiter : str, optional
            The delimiter used in the CSV files (default is ',').

        Returns
        -------
        list
            A list of NumPy arrays corresponding to the processed CSV files.
        """

        arrays = []
        for subject in subjects:
            for file in files:
                array = self._convert_csv_file_to_numpy_array(os.path.join(directory, subject, file), delimiter)
                arrays.append(array)

        return arrays

    @staticmethod
    def _convert_csv_file_to_numpy_array(file_path, delimiter):
        """
        Converts a CSV file into a NumPy array.

        Parameters
        ----------
        file_path : str or pathlib.Path
            The full path to the CSV file.
        delimiter : str
            The delimiter used in the CSV file.

        Returns
        -------
        numpy.ndarray
            A NumPy array representing the data from the CSV file.
        """

        return np.loadtxt(file_path, delimiter=delimiter)

    @staticmethod
    def _calculate_avg_array(numpy_arrays):
        """
        Computes the average of each element across multiple NumPy arrays,
        excluding the highest and lowest 20% of values (to reduce the impact of outliers),
        and returns the resulting array.

        Parameters
        ----------
        numpy_arrays : list
            A list of NumPy arrays to average.

        Returns
        -------
        numpy.ndarray
            A NumPy array containing the average values.
        """

        avg_array = np.zeros(numpy_arrays[0].shape)
        for i in range(numpy_arrays[0].shape[0]):
            for j in range(numpy_arrays[0].shape[1]):

                #ij_values = [numpy_arrays[k][i][j] for k in range(numpy_arrays[0].shape[0])]
                ij_values = [numpy_arrays[k][i][j] for k in range(len(numpy_arrays))]
                # remove highest and lowest X% from ij_values
                ij_values = np.sort(ij_values)
                p = 0.2
                ij_values = ij_values[int(p * len(ij_values)):int((1 - p) * len(ij_values))]

                avg_array[i][j] = np.mean(ij_values)

        # fill diagonal with nans
        # np.fill_diagonal(avg_array, np.nan)

        return avg_array
