# external imports
import numpy as np

# local imports
from simulation.distributions import LagDistributions, DistributionFactory


class DelayCalculator:
    """
    A class to calculate delay distributions based on distance and various statistical parameters.
    The delay is modeled using inverse generalized extreme value (GEV) distributions.

    Attributes
    ----------
    _shape_param : float
        The shape parameter (xi) for the GEV distribution.
    _scale_param : float
        The scale parameter (sigma) for the GEV distribution.
    _location_param : float
        The location parameter (mu) for the GEV distribution.
    _truncation_percentile : float
        The percentile at which to truncate the resulting delay distribution.
    _velocity_factor : float
        A constant velocity factor (= 6) used to calculate the scale coefficient from the distance.
        Expressed in meters per second per micron diameter.
    """

    def __init__(self, shape_param, scale_param, location_param, truncation_percentile):
        """
        Initializes the DelayCalculator with specified parameters for the GEV distribution
        and the truncation percentile.

        Parameters
        ----------
        shape_param : float
            The shape parameter (xi) for the GEV distribution.
        scale_param : float
            The scale parameter (sigma) for the GEV distribution.
        location_param : float
            The location parameter (mu) for the GEV distribution.
        truncation_percentile : float
            The percentile at which to truncate the resulting delay distribution. Must be in the range [0, 1).
        """

        self._shape_param = shape_param
        self._scale_param = scale_param
        self._location_param = location_param
        self._truncation_percentile = truncation_percentile

        self._velocity_factor = 6  # m/s per micron diameter

    def get_delays_distribution(self, tempx, distance):
        """
        Generates a probability density function (PDF) for delays using inverse GEV distributions.
        The distributions are also scaled with parameter computed by :py:meth:`_calculate_scale_coefficient`.

        Depending on whether the distance is a single value or a tuple (in case of a relay station),
        it either sums inverse GEV distributions or uses a single inverse GEV distribution
        (see :py:class:`src.simulation.distributions.LagDistributions` and
        :py:class:`src.simulation.distributions.DistributionFactory`).
        It then truncates the resulting PDF using :py:meth:`_truncate_result`.

        Parameters
        ----------
        tempx : numpy.ndarray
            The array of time points (x-axis) over which to calculate the delay distribution.
        distance : float or tuple
            The distance(s) over which to calculate the delay. If a tuple, the method will use
            the sum of two inverse GEV distributions.

        Returns
        -------
        numpy.ndarray
            The truncated probability density function (PDF) representing the delay distribution.

        Raises
        ------
        AssertionError
            If the distribution cannot be created.
        """

        distribution = None

        if isinstance(distance, tuple):
            # we use sum of inverse GEVs
            scale_coef1 = self._calculate_scale_coefficient(distance[0])
            scale_coef2 = self._calculate_scale_coefficient(distance[1])
            distribution = DistributionFactory.get_distribution(
                LagDistributions.INVERSE_GEV_SUM,
                lmbd1=scale_coef1,
                lmbd2=scale_coef2,
                mu=self._location_param,
                sigma=self._scale_param,
                xi=self._shape_param
            )

        elif isinstance(distance, float):
            # we use inverse GEV
            scale_coef = self._calculate_scale_coefficient(distance)
            distribution = DistributionFactory.get_distribution(
                LagDistributions.INVERSE_GEV,
                lmbd=scale_coef,
                mu=self._location_param,
                sigma=self._scale_param,
                xi=self._shape_param
            )

        assert distribution is not None, "Couldn't create distribution"
        delays_pdf = distribution.pdf(tempx)

        return self._truncate_result(tempx, delays_pdf)

    def _calculate_scale_coefficient(self, distance):
        """
        Calculates the scale coefficient for the GEV distribution based on the given distance.

        Parameters
        ----------
        distance : float
            The distance for which to calculate the scale coefficient.

        Returns
        -------
        float
            The scale coefficient used in the GEV distribution.
        """

        return distance / self._velocity_factor
        # return self.velocity_factor / (1 * distance)

    def _truncate_result(self, tempx, result):
        """
        Truncates the PDF by setting values beyond a certain index to zero, based on the cumulative
        distribution function (CDF) and the truncation percentile.

        Parameters
        ----------
        tempx : numpy.ndarray
            The array of time points (x-axis) corresponding to the PDF.
        result : numpy.ndarray
            The PDF to be truncated.

        Returns
        -------
        numpy.ndarray
            The truncated PDF.

        Raises
        ------
        AssertionError
            If the truncation percentile is outside the valid range [0, 1).
        """

        assert 0 <= self._truncation_percentile < 1, "Truncation percentile must be in the range [0, 1)"

        if self._truncation_percentile == 0:
            return result

        # Calculate the cumulative sum of the PDF to simulate the CDF
        pdf_sum = np.sum(result)

        # Substitute with a safe value if pdf_sum is zero
        if pdf_sum == 0:
            cdf = np.zeros_like(result)
        else:
            cdf = np.cumsum(result) / pdf_sum
        # Find the maximum index where the cumulative sum is less than or equal to 0.04 (4%)
        valid_indices = np.where(cdf <= (1 - self._truncation_percentile))[0]
        if valid_indices.size > 0:
            cutoff_index = np.max(valid_indices) if np.any(cdf <= 0.04) else 0
        else:
            cutoff_index = 0
        # Truncate values in the PDF beyond this index
        truncated_result = np.where(tempx > tempx[cutoff_index], 0, result)

        return truncated_result

