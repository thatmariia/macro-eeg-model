macro_eeg_model.evaluation.peak_tester
======================================

.. py:module:: macro_eeg_model.evaluation.peak_tester


Classes
-------

.. autoapisummary::

   macro_eeg_model.evaluation.peak_tester.PeakTester


Module Contents
---------------

.. py:class:: PeakTester(frequencies, peaks_range, others_range)

   A class responsible for testing the significance of peak power values compared to other frequency ranges.

   .. attribute:: frequencies

      The array of frequencies corresponding to the power spectrum.

      :type: numpy.ndarray

   .. attribute:: peaks_range

      The range of frequencies where peaks are expected.

      :type: tuple

   .. attribute:: others_range

      The range of frequencies where other values are expected.

      :type: tuple

   .. attribute:: powers

      The epoched power spectrum of the simulated EEG data.

      :type: list

   .. attribute:: peak_values

      The mean power values in the peak range for each epoch.

      :type: list

   .. attribute:: other_values

      The mean power values in the other range for each epoch.

      :type: list


   .. py:method:: __init__(frequencies, peaks_range, others_range)

      Initializes the PeakTester class with the provided frequency ranges.

      :param frequencies: The array of frequencies corresponding to the power spectrum.
      :type frequencies: numpy.ndarray
      :param peaks_range: The range of frequencies where peaks are expected.
      :type peaks_range: tuple
      :param others_range: The range of frequencies where other values are expected.
      :type others_range: tuple



   .. py:method:: compute_test_result(simulation_name, epoched_powers)

      Computes the statistical test result for the peak power values compared to other frequency ranges.

      :param simulation_name: The name of the simulation. (should include "pink" if the data was simulated with pink noise)
      :type simulation_name: str
      :param epoched_powers: The epoched power spectrum of the simulated EEG data.
      :type epoched_powers: list

      :returns: A tuple containing:

                - frequencies (numpy.ndarray): The array of frequencies corresponding to the power spectrum.
                - mean_power (numpy.ndarray): The mean power spectrum across epochs of the simulated EEG data.
                - p_value (float): The calculated p-value.
                - test_name (str): The name of the statistical test used.
      :rtype: tuple



   .. py:method:: _separate_peaks(power)

      Separates the power values in the peak and other frequency ranges.

      :param power: The power spectrum of the simulated EEG data.
      :type power: numpy.ndarray



   .. py:method:: _detrend_data(powers, is_pink)

      Detrend the pink noise in the power spectrum by fitting a power-law trend and removing it.

      :param powers: The power spectrum of the simulated EEG data.
      :type powers: numpy.ndarray
      :param is_pink: A flag indicating whether the data was simulated with pink noise.
      :type is_pink: bool

      :returns: A tuple containing:

                - non_zero_freqs (numpy.ndarray): The array of non-zero frequencies.
                - flattened_powers (numpy.ndarray): The corresponding detrended power spectrum
      :rtype: tuple



   .. py:method:: _choose_and_run_test(paired=True)

      Automatically selects and runs the correct statistical test based on data characteristics.

      :param paired: A flag indicating whether the data is paired or independent.
      :type paired: bool

      :returns: A tuple containing:

                - t_stat (float): The calculated t-statistic.
                - p_value (float): The calculated p-value.
                - test_name (str): The name of the statistical test used.
      :rtype: tuple



