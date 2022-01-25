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
    Implements Agresti-Coull confidence interval for binary outcomes.

    Args:
        x (array): Number of successes observed
        n (array): Number of trials
        conf (array): Confidence level for the interval

    Returns:
        dataframe: Pandas dataframe containing the estimate and bounds for the interval

    .. note::
        Though the Agresti-Coull interval uses :math:`\\tilde{p}` as the center of the interval, pynomial returns :math:`p=x/n` as an unbiased estimate of the risk.

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

    Args:
        x (array): Number of successes observed
        n (array): Number of trials
        conf (array): Confidence level for the interval

    Returns:
        dataframe: Pandas dataframe containing the estimate and bounds for the interval

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

       Args:
        x (array): Number of successes observed
        n (array): Number of trials
        conf (array): Confidence level for the interval
        shape_1 (float): First shape parameter for the prior distribution
        shape_2 (float): Second shape parameter for the prior distirbution

    Returns:
        dataframe: Pandas dataframe containing the estimate and bounds for the interval
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
     Implements complimentary log-log interval for binary outcomes.

    Args:
        x (array): Number of successes observed
        n (array): Number of trials
        conf (array): Confidence level for the interval

    Returns:
        dataframe: Pandas dataframe containing the estimate and bounds for the interval
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

    Args:
        x (array): Number of successes observed
        n (array): Number of trials
        conf (array): Confidence level for the interval

    Returns:
        dataframe: Pandas dataframe containing the estimate and bounds for the interval
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
    Implements exact confidence intervals using quantiles of the beta distirbution.

      Args:
        x (array): Number of successes observed
        n (array): Number of trials
        conf (array): Confidence level for the interval

    Returns:
        dataframe: Pandas dataframe containing the estimate and bounds for the interval
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

    Args:
        x (array): Number of successes observed
        n (array): Number of trials
        conf (array): Confidence level for the interval

    Returns:
        dataframe: Pandas dataframe containing the estimate and bounds for the interval

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

    Args:
        x (array): Number of successes observed
        n (array): Number of trials
        conf (array): Confidence level for the interval
        *args: Not used
        **kwargs: Keyword arguments passed to root finding procedure

    Returns:
        dataframe: Pandas dataframe containing the estimate and bounds for the interval

    """

    x, n, conf = _check_args(x, n, conf)
    p = x / n
    lower, upper = _lrt_rootfinding(x, n, conf, *args, **kwargs)

    interval = _interval(estimate=p, lower=lower, upper=upper, method="LRT")

    return interval
