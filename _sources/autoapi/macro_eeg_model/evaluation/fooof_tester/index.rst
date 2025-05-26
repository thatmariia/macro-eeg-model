macro_eeg_model.evaluation.fooof_tester
=======================================

.. py:module:: macro_eeg_model.evaluation.fooof_tester


Classes
-------

.. autoapisummary::

   macro_eeg_model.evaluation.fooof_tester.FooofTester


Module Contents
---------------

.. py:class:: FooofTester(frequencies)

   A class responsible for testing the presence of peaks in EEG power spectra using FOOOF.

   .. attribute:: frequencies

      The array of frequencies corresponding to the power spectrum.

      :type: numpy.ndarray

   .. attribute:: fm

      The FOOOF object for fitting spectral models.

      :type: FOOOF


   .. py:method:: __init__(frequencies)

      Initializes the FooofTester class using FOOOF.

       Parameters
       ----------
       frequencies : numpy.ndarray
           The array of frequencies corresponding to the power spectrum.




   .. py:method:: get_peaks_positions(powers)

      Extracts the peak positions from the power spectrum using FOOOF.

      :param powers: The power spectrum from which to extract peaks.
      :type powers: numpy.ndarray

      :returns: A binary array indicating the presence of peaks at the specified frequencies.
                1 if a peak is present at a given frequency, 0 otherwise.
      :rtype: numpy.ndarray



