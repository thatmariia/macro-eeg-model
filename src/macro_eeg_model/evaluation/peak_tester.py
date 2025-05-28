# external imports
from scipy.stats import ttest_rel, wilcoxon, shapiro, ttest_ind, levene, mannwhitneyu
import numpy as np
from scipy.optimize import curve_fit


class PeakTester:
    """
    A class responsible for testing the significance of peak power values compared to other frequency ranges.

    Attributes
    ----------
    frequencies : numpy.ndarray
        The array of frequencies corresponding to the power spectrum.
    peaks_range : tuple
        The range of frequencies where peaks are expected.
    others_range : tuple
        The range of frequencies where other values are expected.
    powers : list
        The epoched power spectrum of the simulated EEG data.
    peak_values : list
        The mean power values in the peak range for each epoch.
    other_values : list
        The mean power values in the other range for each epoch.
    """

    def __init__(self, frequencies, peaks_range, others_range):
        """
        Initializes the PeakTester class with the provided frequency ranges.

        Parameters
        ----------
        frequencies : numpy.ndarray
            The array of frequencies corresponding to the power spectrum.
        peaks_range : tuple
            The range of frequencies where peaks are expected.
        others_range : tuple
            The range of frequencies where other values are expected.
        """

        self.frequencies = frequencies
        self.peaks_range = peaks_range
        self.others_range = others_range
        self.powers = []

        self.peak_values = []
        self.other_values = []

    def compute_test_result(self, simulation_name, epoched_powers):
        """
        Computes the statistical test result for the peak power values compared to other frequency ranges.

        Parameters
        ----------
        simulation_name : str
            The name of the simulation. (should include "pink" if the data was simulated with pink noise)
        epoched_powers : list
            The epoched power spectrum of the simulated EEG data.

        Returns
        -------
        tuple
            A tuple containing:

            - frequencies (numpy.ndarray): The array of frequencies corresponding to the power spectrum.
            - mean_power (numpy.ndarray): The mean power spectrum across epochs of the simulated EEG data.
            - p_value (float): The calculated p-value.
            - test_name (str): The name of the statistical test used.
        """

        frequencies = None
        for epoched_power in epoched_powers:
            frequencies, detrended_power = self._detrend_data(epoched_power, is_pink="pink" in simulation_name)
            self.powers.append(detrended_power)

            mean_peaks, mean_others = self._separate_peaks(detrended_power)
            self.peak_values.append(mean_peaks)
            self.other_values.append(mean_others)

        t_stat, p_value, test_name = self._choose_and_run_test(paired=True)

        return frequencies, np.mean(self.powers, axis=0), p_value, test_name

    def _separate_peaks(self, power):
        """
        Separates the power values in the peak and other frequency ranges.

        Parameters
        ----------
        power : numpy.ndarray
            The power spectrum of the simulated EEG data.
        """

        peak_values = []
        other_values = []

        for i, freq in enumerate(self.frequencies):
            if self.peaks_range[0] <= freq <= self.peaks_range[1]:
                peak_values.append(power[i])
            elif self.others_range[0] < freq <= self.others_range[1]:
                other_values.append(power[i])

        return np.mean(peak_values), np.mean(other_values)

    def _detrend_data(self, powers, is_pink):
        """
        Detrend the pink noise in the power spectrum by fitting a power-law trend and removing it.

        Parameters
        ----------
        powers : numpy.ndarray
            The power spectrum of the simulated EEG data.
        is_pink : bool
            A flag indicating whether the data was simulated with pink noise.

        Returns
        -------
        tuple
            A tuple containing:

            - non_zero_freqs (numpy.ndarray): The array of non-zero frequencies.
            - flattened_powers (numpy.ndarray): The corresponding detrended power spectrum
        """

        non_zero_freqs = self.frequencies[1:]
        non_zero_powers = powers[1:len(self.frequencies)]

        if not is_pink:
            return non_zero_freqs, non_zero_powers

        def power_law(f, alpha, C):
            return C * f ** (-alpha)

        # Fit the power spectrum to a power-law trend
        popt, _ = curve_fit(power_law, non_zero_freqs, non_zero_powers, p0=[1.0, 1.0], maxfev=5000)
        alpha_fit, C_fit = popt

        # Get a "white noise-like" version
        trend = power_law(non_zero_freqs, alpha_fit, C_fit)
        flattened_powers = non_zero_powers / (1 / non_zero_freqs)  # trend + np.mean(non_zero_powers)  #/ trend
        scale_factor = np.sqrt(np.sum(non_zero_powers ** 2) / np.sum(flattened_powers ** 2))
        flattened_powers *= scale_factor

        return non_zero_freqs, flattened_powers

    def _choose_and_run_test(self, paired=True):
        """
        Automatically selects and runs the correct statistical test based on data characteristics.

        Parameters
        ----------
        paired : bool
            A flag indicating whether the data is paired or independent.

        Returns
        -------
        tuple
            A tuple containing:

            - t_stat (float): The calculated t-statistic.
            - p_value (float): The calculated p-value.
            - test_name (str): The name of the statistical test used.
        """

        if paired:
            # Check normality of differences (Shapiro-Wilk test)
            diffs = [a - b for a, b in zip(self.peak_values, self.other_values)]
            p_value_normality = shapiro(diffs).pvalue
            # print(f"Shapiro-Wilk test on differences: p = {p_value_normality:.4f}")

            if p_value_normality > 0.05:
                # Paired t-test (parametric)
                test_name = "Paired t-test"
                t_stat, p_value = ttest_rel(self.peak_values, self.other_values, alternative='greater')
            else:
                # Wilcoxon signed-rank test (non-parametric)
                test_name = "Wilcoxon test"
                t_stat, p_value = wilcoxon(self.peak_values, self.other_values, alternative='greater')

        else:
            # Check normality of both lists (Shapiro-Wilk test)
            p_norm1 = shapiro(self.peak_values).pvalue
            p_norm2 = shapiro(self.other_values).pvalue
            # print(f"Shapiro-Wilk test: Peaks p = {p_norm1:.4f}, Others p = {p_norm2:.4f}")

            if p_norm1 > 0.05 and p_norm2 > 0.05:
                # Check variance homogeneity (Levene's test)
                p_var = levene(self.peak_values, self.other_values).pvalue
                # print(f"Levene’s test for equal variances: p = {p_var:.4f}")

                if p_var > 0.05:
                    # Independent t-test (parametric, equal variances)
                    test_name = "Ind. t-test"
                    t_stat, p_value = ttest_ind(self.peak_values, self.other_values, alternative='greater')
                else:
                    # Welch's t-test (parametric, unequal variances)
                    test_name = "Welch's t-test"
                    t_stat, p_value = ttest_ind(self.peak_values, self.other_values, equal_var=False, alternative='greater')
            else:
                # Mann-Whitney U test (non-parametric)
                test_name = "Mann-Whitney U test"
                t_stat, p_value = mannwhitneyu(self.peak_values, self.other_values, alternative='greater')

        return t_stat, p_value, test_name
