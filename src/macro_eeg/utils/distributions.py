# delay_distributions.py
from enum import Enum
import numpy as np
from scipy.stats import rv_continuous
from sklearn.neighbors import KernelDensity
from typing import Literal


def get_distribution(dist_type: Literal["inv_gev", "inv_gev_sum"], **kw) -> rv_continuous:
        if dist_type == "inv_gev":
            return InverseGEV(
                lmbd=kw["lmbd"],
                mu=kw["mu"],
                sigma=kw["sigma"],
                xi=kw["xi"],
            )
        if dist_type == "inv_gev_sum":
            return InverseGEVSum(
                lmbd1=kw["lmbd1"],
                lmbd2=kw["lmbd2"],
                mu=kw["mu"],
                sigma=kw["sigma"],
                xi=kw["xi"],
            )
        raise ValueError(f"unknown distribution: {dist_type}")


class InverseGEV(rv_continuous):
    """
    A class representing the inverse generalized extreme value (GEV) distribution.

    This class extends `scipy.stats.rv_continuous` to model the inverse GEV distribution.

    Attributes
    ----------
    lmbd : float
        A scaling parameter applied to the distribution.
    mu : float
        The location parameter of the GEV distribution.
    sigma : float
        The scale parameter of the GEV distribution.
    xi : float
        The shape parameter of the GEV distribution.
    """

    def __init__(self, lmbd, mu, sigma, xi, *args, **kwargs):
        """
        Initializes the InverseGEV distribution with the specified parameters.

        Parameters
        ----------
        lmbd : float
            A scaling parameter applied to the distribution.
        mu : float
            The location parameter of the GEV distribution.
        sigma : float
            The scale parameter of the GEV distribution.
        xi : float
            The shape parameter of the GEV distribution.
        """

        super().__init__(*args, **kwargs)
        self.lmbd = lmbd
        self.mu = mu  # location
        self.sigma = sigma  # scale
        self.xi = xi  # shape

    def _argcheck(self, *args):
        """
        Validates the distribution parameters.

        Returns
        -------
        bool
            True if the parameters are valid, False otherwise.
        """
        return bool((self.sigma > 0) and (self.xi != 0))

    def _cdf(self, x, *args):
        """
        Calculates the cumulative distribution function (CDF) for the inverse GEV.

        Parameters
        ----------
        x : array_like
            The quantiles at which to evaluate the CDF.

        Returns
        -------
        array_like
            The CDF evaluated at the given quantiles.
        """

        def gev_cdf(n):
            s = (n - self.mu) / self.sigma
            t = 1 + self.xi * s
            term = np.power(t, -1 / self.xi)
            return np.exp(-1 * term)

        return 1 - gev_cdf(self.lmbd / x)

    def _pdf(self, x, *args):
        """
        Calculates the probability density function (PDF) for the inverse GEV.

        Parameters
        ----------
        x : array_like
            The quantiles at which to evaluate the PDF.

        Returns
        -------
        array_like
            The PDF evaluated at the given quantiles.
        """

        t = 1 + self.xi * ((self.lmbd - self.mu * x) / (self.sigma * x))
        t = np.where(t < 0, 0, t)
        t = np.array(
            [
                0 if ((ti == 0) and (-1 / self.xi < 0)) else np.power(ti, -1 / self.xi)
                for ti in t
            ]
        )

        term1 = np.array(
            [
                0 if ((ti == 0) and (self.xi + 1 < 0)) else np.power(ti, self.xi + 1)
                for ti in t
            ]
        )

        term2 = np.exp(-1 * t)
        pdf = (self.lmbd / np.power(x, 2)) * (1 / self.sigma) * term1 * term2
        pdf = np.where(pdf < 0, 0, pdf)

        return pdf

    def _ppf(self, q, *args):
        """
        Calculates the percent point function (PPF), also known as the quantile function, for the inverse GEV.

        Parameters
        ----------
        q : array_like
            The quantiles for which to evaluate the PPF.

        Returns
        -------
        array_like
            The PPF evaluated at the given quantiles.
        """

        # Inverse CDF (quantile function)
        def gev_ppf(p):
            term = -np.log(p)
            t = np.power(term, -self.xi)
            s = (t - 1) / self.xi
            return self.mu + self.sigma * s

        return self.lmbd / gev_ppf(1 - q)

    def _rvs(self, *args, size=None, random_state=None):
        """
        Generates random variates from the inverse GEV distribution.

        Parameters
        ----------
        size : int or tuple of ints, optional
            The number of random variates to generate.
        random_state : np.random.RandomState, optional
            A random state instance for reproducibility.

        Returns
        -------
        array_like
            The generated random variates.
        """
        u = self._random_state.random_sample(size)
        return self._ppf(u)


class InverseGEVSum(rv_continuous):
    """
    A class representing the sum of two inverse GEV distributions.

    This class extends `scipy.stats.rv_continuous` to model the sum of two inverse GEV distributions.
    The sum is approximated using a kernel density estimate (KDE) of the sum of samples from the two distributions.

    Attributes
    ----------
    lmbd1 : float
        A scaling parameter for the first inverse GEV distribution.
    lmbd2 : float
        A scaling parameter for the second inverse GEV distribution.
    mu : float
        The location parameter of the GEV distributions.
    sigma : float
        The scale parameter of the GEV distributions.
    xi : float
        The shape parameter of the GEV distributions.
    _kde : KernelDensity
        The kernel density estimate of the sum of the two inverse GEV distributions.
    """

    def __init__(self, lmbd1, lmbd2, mu, sigma, xi, *args, **kwargs):
        """
        Initializes the InverseGEVSum distribution with the specified parameters.

        Parameters
        ----------
        lmbd1 : float
            A scaling parameter for the first inverse GEV distribution.
        lmbd2 : float
            A scaling parameter for the second inverse GEV distribution.
        mu : float
            The location parameter of the GEV distributions.
        sigma : float
            The scale parameter of the GEV distributions.
        xi : float
            The shape parameter of the GEV distributions.
        """

        super().__init__(*args, **kwargs)
        self.lmbd1 = lmbd1
        self.lmbd2 = lmbd2
        self.mu = mu
        self.sigma = sigma
        self.xi = xi

        self._kde = self._get_kde()

    def _get_kde(self):
        """
        Generates a kernel density estimate (KDE) for the sum of two inverse GEV distributions.

        Returns
        -------
        KernelDensity
            A KDE fitted to the sum of samples from the two inverse GEV distributions.
        """

        inverse_gev1 = InverseGEV(
            lmbd=self.lmbd1, mu=self.mu, sigma=self.sigma, xi=self.xi
        )
        inverse_gev2 = InverseGEV(
            lmbd=self.lmbd2, mu=self.mu, sigma=self.sigma, xi=self.xi
        )

        # generate samples for 1 and 2
        samples1 = inverse_gev1.rvs(size=1000)
        samples2 = inverse_gev2.rvs(size=1000)

        samples = samples1 + samples2

        silverman_bandwidth = (4 * np.std(samples) ** 5 / (3 * len(samples))) ** (1 / 5)
        return KernelDensity(kernel="gaussian", bandwidth=silverman_bandwidth).fit(
            samples[:, np.newaxis]
        )

    def _argcheck(self, *args):
        """
        Validates the distribution parameters.

        Returns
        -------
        bool
            True if the parameters are valid, False otherwise.
        """

        return bool((self.sigma > 0) and (self.xi != 0))

    def _pdf(self, x, *args):
        """
        Evaluates the kernel density estimate (KDE) at the given quantiles to
        approximate the probability density function (PDF) for the sum of inverse GEVs.

        Parameters
        ----------
        x : array_like
            The quantiles at which to evaluate the PDF.

        Returns
        -------
        array_like
            The PDF evaluated at the given quantiles.
        """

        return np.exp(self._kde.score_samples(x[:, np.newaxis]))
    def __init__(
        self,
        lmbd1: float,
        lmbd2: float,
        mu: float,
        sigma: float,
        xi: float,
        *,
        n_samples: int = 1000,
        rng: np.random.Generator | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.lmbd1 = float(lmbd1)
        self.lmbd2 = float(lmbd2)
        self.mu = float(mu)
        self.sigma = float(sigma)
        self.xi = float(xi)
        self.n_samples = n_samples
        self.rng = rng or np.random.default_rng()
        self._kde = self._build_kde()

    def _build_kde(self) -> KernelDensity:
        inv1 = InverseGEV(self.lmbd1, self.mu, self.sigma, self.xi)
        inv2 = InverseGEV(self.lmbd2, self.mu, self.sigma, self.xi)

        # use ppf with uniform samples from our own rng
        u1 = self.rng.random(self.n_samples)
        u2 = self.rng.random(self.n_samples)

        x1 = inv1._ppf(u1)
        x2 = inv2._ppf(u2)
        samples = x1 + x2

        # Silverman's rule
        std = np.std(samples)
        bw = (4 * std**5 / (3 * len(samples))) ** (1 / 5) if std > 0 else 0.1
        return KernelDensity(kernel="gaussian", bandwidth=bw).fit(samples[:, None])

    def _argcheck(self, *args):
        return bool(self.sigma > 0 and self.xi != 0)

    def _pdf(self, x, *args):
        x = np.asarray(x, dtype=float)
        return np.exp(self._kde.score_samples(x[:, None]))