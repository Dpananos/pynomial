import numpy as np
import pandas as pd
from scipy.stats import norm, beta, binom, chi2
from scipy.optimize import newton
from scipy.special import expit, logit as log_it
from .utils import _check_args, _lrt_rootfinding

__all__ = [
    "agresti_coull",
    "asymptotic",
    "bayes",
    "loglog",
    "wilson",
    "exact",
    "logit",
    "lrt",
]


def _interval(estimate, lower, upper, method):

    if hasattr(estimate, "__len__"):
        index = len(estimate) * [method]
    else:
        index = [method]

    interval_df = pd.DataFrame(
        {"estimate": estimate, "lower": lower, "upper": upper}, index=index
    )

    return interval_df


def agresti_coull(x, n, conf=0.95, *args, **kwargs):

    """
    Implements Agresti-Coull confidence interval for binary outcomes. Given :math:`x` successes in :math:`n` trials, define

    .. math::
        \\tilde{n}=n+z^{2}

    and

    .. math::
        \\tilde{p}=\\frac{1}{\widetilde{n}}\left(x+\\frac{z^{2}}{2}\\right)

    The Agresti-Coul interval is then

    .. math::
        \\tilde{p} \pm z \sqrt{\\tilde{p}(1-\\tilde{p}) \over \\tilde{n}}

    Here, :math:`z` is the :math:`1-\\alpha` quantile of a standard normal distribution


    .. note::
        Though the Agresti-Coull interval uses :math:`\\tilde{p}` as the center of the interval, pynomial returns :math:`p=x/n` as an unbiased estimate of the risk


    References:

    1. Agresti, Alan, and Brent A. Coull. “Approximate Is Better than ‘Exact’ for Interval Estimation of Binomial Proportions.” The American Statistician, vol. 52, no. 2, [American Statistical Association, Taylor & Francis, Ltd.], 1998, pp. 119–26, https://doi.org/10.2307/2685469.
    """

    x, n, conf = _check_args(x, n, conf)

    alpha = 1 - conf
    z = norm().ppf(1 - alpha / 2)
    p = x / n
    ntilde = n + np.power(z, 2)
    ptilde = (x + np.power(z, 2) / 2) / ntilde

    upper = ptilde + z * np.sqrt(ptilde * (1 - ptilde) / ntilde)
    lower = ptilde - z * np.sqrt(ptilde * (1 - ptilde) / ntilde)

    interval = _interval(estimate=p, lower=lower, upper=upper, method="Agresti-Coull")

    return interval


def asymptotic(x, n, conf=0.95, *args, **kwargs):

    """
    Implements asymptotic confidence interval for binary outocmes derived from the central limit theorem.

    Given :math:`x` successes in `n` trials, the estimate of the risk is :math:`p=x/n` and the confidence interval is

    .. math::
        p \pm z \sqrt{p(1-p) \over n}

    Here, :math:`z` is the :math:`1-\\alpha` quantile of a standard normal distribution


    References:

    1. Lachin, John M. Biostatistical methods: the assessment of relative risks. Vol. 509. John Wiley & Sons, 2009.

    """

    x, n, conf = _check_args(x, n, conf)

    alpha = 1 - conf
    z = norm().ppf(1 - alpha / 2)
    p = x / n
    asymptotic_var = p * (1 - p) / n
    lower = p - z * np.sqrt(asymptotic_var)
    upper = p + z * np.sqrt(asymptotic_var)

    interval = _interval(estimate=p, lower=lower, upper=upper, method="Asymptotic")

    return interval


def bayes(x, n, conf=0.95, shape_1=0.5, shape_2=0.5, *args, **kwargs):

    """
    Implements a Bayesian model with beta prior on the risk parameter.  Resulting confidence interval is actually an equal tailed posterior credible interval.

    Using the prior

    .. math::
        P(p) = \operatorname{Beta}(a, b)

    The posterior of :math:`p` given :math:`x,n` is

    .. math::
        P(p \\vert x, n ) = \operatorname{Beta}(a + x, b + n - x)


    The posterior mean is estimated as :math:`(a+x)/(a+b+n)`. The confidence interval is obtained by computing the :math:`\\alpha` and :math:`1-\\alpha` quantiles of the posterior.

    .. note::

        Whereas the binom library implements central and highest density credible intervals, pynomial only
        implements a central inverval.

    References:

    1. Svensén, Markus, and Christopher M. Bishop. "Pattern recognition and machine learning." (2007).
    """

    x, n, conf = _check_args(x, n, conf)

    if shape_1 < 0 or shape_2 < 0:
        raise ValueError("shape parameters must be positive reals")

    alpha = 1 - conf
    a = shape_1 + x
    b = shape_2 + n - x

    posterior = beta(a=a, b=b)

    p = a / (a + b)
    lower = posterior.ppf(alpha / 2)
    upper = posterior.ppf(1 - alpha / 2)

    interval = _interval(estimate=p, lower=lower, upper=upper, method="Bayes")

    return interval


def loglog(x, n, conf=0.95, *args, **kwargs):

    """
     Implements complimentary log-log interval for binary outcomes. Given :math:`x` successes in :math:`n` trials, define

     .. math::
         \\theta = g(p) = \log(-\log(p))

     The confidence interval on the loglog scale is

     .. math::
         \left(\\theta_{L}, \\theta_{U} \\right) = \\theta \pm z \sqrt{\\frac{(1-p)}{n p(\log p)^{2}}}

    Here, :math:`z` is the :math:`1-\\alpha` quantile of a standard normal distribution. Applying :math:`g^{-1}` to each endpoint yields a confidence interval on the original scale.

    .. math::

     \left(p_L , p_R \\right)= \left(\exp \left[ -\exp \left(\\theta_{U} \\right) \\right], \exp \left[ -\exp \left(\\theta_{L} \\right) \\right] \\right)

    """

    x, n, conf = _check_args(x, n, conf)

    alpha = 1 - conf
    z = norm().ppf(1 - alpha / 2)
    p = x / n
    theta = np.log(-np.log(p))
    radius = z * np.sqrt((1 - p) / (n * p * np.log(p) ** 2))
    theta_lower = theta - radius
    theta_upper = theta + radius

    # This is not a mistake.  The end points on the log-log scale are permuted with the endpoints
    # On the natural scale.  See pg. 17 of Lachin 2ed, equation 2.19
    upper = np.exp(-np.exp(theta_lower))
    lower = np.exp(-np.exp(theta_upper))

    interval = _interval(estimate=p, lower=lower, upper=upper, method="Cloglog")

    return interval


def wilson(x, n, conf=0.95, *args, **kwargs):

    """
    Implements Wilson score interval for binary outcomes.

        Parameters:
            x (int or array): An integer or array of integers for the number of positive outcomes

            n (int or array): An integer or array of integers for the number of trials

            conf (float): The confidence level desired

        Returns:
            interval (dataframe):  A dataframe housing the risk estimate and upper/lower confidence bounds
    """

    x, n, conf = _check_args(x, n, conf)

    alpha = 1 - conf
    z = norm().ppf(1 - alpha / 2)

    p = x / n
    z2 = np.power(z, 2)
    center = 1.0 / (1 + z2 / n) * (p + z2 / (2 * n))
    radius = z / (1 + z2 / n) * np.sqrt(p * (1 - p) / n + z2 / (4 * np.power(n, 2)))
    lower = center - radius
    upper = center + radius

    interval = _interval(estimate=p, lower=lower, upper=upper, method="Wilson")

    return interval


def exact(x, n, conf=0.95, *args, **kwargs):

    """
    Implements exact confidence intervals for the binomial using results from the incomplete beta function.

        Parameters:
            x (int or array): An integer or array of integers for the number of positive outcomes

            n (int or array): An integer or array of integers for the number of trials

            conf (float): The confidence level desired

        Returns:
            interval (dataframe):  A dataframe housing the risk estimate and upper/lower confidence bounds
    """

    x, n, conf = _check_args(x, n, conf)

    alpha = 1 - conf
    p = x / n
    lower = 1 - beta(a=n + 1 - x, b=x).ppf(1 - alpha / 2)
    upper = 1 - beta(a=n - x, b=x + 1).ppf(alpha / 2)
    interval = _interval(estimate=p, lower=lower, upper=upper, method="Exact")

    return interval


def logit(x, n, conf=0.95, *args, **kwargs):
    """
    Implements logit confidence intervals for the binomial using large sample variance from the Delta method.

        Parameters:
            x (int or array): An integer or array of integers for the number of positive outcomes

            n (int or array): An integer or array of integers for the number of trials

            conf (float): The confidence level desired

    """

    x, n, conf = _check_args(x, n, conf)

    alpha = 1 - conf
    z = norm().ppf(1 - alpha / 2)
    p = x / n
    logit_p = log_it(p)
    logit_var = 1 / (n * p * (1 - p))
    radius = z * np.sqrt(logit_var)
    theta_upper = logit_p + radius
    theta_lower = logit_p - radius
    lower = expit(theta_lower)
    upper = expit(theta_upper)
    interval = _interval(estimate=p, lower=lower, upper=upper, method="Logit")

    return interval


def lrt(x, n, conf=0.95, *args, **kwargs):

    """
    Implements confidence intervals by inverting the Likelihood Ratio Test (LRT).  This method
    uses root finding procedures via scipy.optimize.newton. Keyword arguments to the
    root finding algorithm can be passed via .

    The LRT for a binomial risk parameter is

    .. math::
        \ell( \hat{\\theta}, \\theta^\star) =  -2 \left( x\log\left( \hat{\\theta} \over \\theta^{\star} \\right) + (n-x)\log\left( (1-\hat{\\theta}) \over (1-{\\theta}^\star)  \\right) \\right)

    Here, :math:`\hat{\\theta}` is the estimated risk and :math:`\\theta^{\star}` is the risk under the null hypothesis.  To create confidence intervals, this test is inverted to solve for the two roots of the equation

    .. math::
        \ell( \hat{\\theta}, \\theta^\star) - \chi^2_{1-\\alpha} = 0

    Where  :math:`\chi^2_{1-\\alpha}` is the critical value for the LRT.

    """

    x, n, conf = _check_args(x, n, conf)
    p = x / n
    lower, upper = _lrt_rootfinding(x, n, conf)

    interval = _interval(estimate=p, lower=lower, upper=upper, method="LRT")

    return interval
