import numpy as np
import pandas as pd
from scipy.stats import norm, beta, binom
from scipy.special import expit, logit
from .utils import _check_arguments


def _interval(estimate, lower, upper, method):

    if hasattr(estimate, "__len__"):
        index = len(estimate) * [method]
    else:
        index = [method]

    interval_df = pd.DataFrame(
        {"estimate": estimate, "lower": lower, "upper": upper}, index=index
    )

    return interval_df


def agresti_coull(x, n, conf=0.95):

    """
    Implements Agresti-Coull confidence interval for binary outcomes.

        Parameters:
            x (int or array): An integer or array of integers for the number of positive outcomes
            n (int or array): An integer or array of integers for the number of trials
            conf (float): The confidence level desired

        Returns:
            interval (dataframe):  A dataframe housing the risk estimate and upper/lower confidence bounds
    """

    _check_arguments(x, n, conf)

    alpha = 1 - conf
    z = norm().ppf(1 - alpha / 2)
    p = x / n
    ntilde = n + np.power(z, 2)
    ptilde = (x + np.power(z, 2) / 2) / ntilde

    upper = ptilde + z * np.sqrt(ptilde * (1 - ptilde) / ntilde)
    lower = ptilde - z * np.sqrt(ptilde * (1 - ptilde) / ntilde)

    interval = _interval(estimate=p, lower=lower, upper=upper, method="Agresti Coull")

    return interval


def asymptotic(x, n, conf=0.95):

    """
    Implements asymptotic confidence interval for binary outocmes derived from the central limit theorem.

        Parameters:
            x (int or array): An integer or array of integers for the number of positive outcomes
            n (int or array): An integer or array of integers for the number of trials
            conf (float): The confidence level desired

        Returns:
            interval (dataframe):  A dataframe housing the risk estimate and upper/lower confidence bounds
    """

    alpha = 1 - conf
    z = norm().ppf(1 - alpha / 2)
    p = x / n
    asymptotic_var = p * (1 - p) / n
    lower = p - z * np.sqrt(asymptotic_var)
    upper = p + z * np.sqrt(asymptotic_var)

    interval = _interval(estimate=p, lower=lower, upper=upper, method="Asymptotic")

    return interval


def bayes(x, n, conf=0.95, shape_1=0.5, shape_2=0.5):

    """
    Implements a Bayesian model with beta prior on the risk parameter.  Resulting confidence interval is actually an equal tailed posterior credible interval.

        Parameters:
            x (int or array): An integer or array of integers for the number of positive outcomes
            n (int or array): An integer or array of integers for the number of trials
            conf (float): The confidence level desired
            shape_1 (float): First shape parameter for the beta prior on the risk parameter.  Must be a positive float.
            shape_2 (float): Second shape parameter for the beta prior on the risk parameter.  Must be a positive float.

        Returns:
            interval (dataframe):  A dataframe housing the risk estimate and upper/lower confidence bounds
    """

    _check_arguments(x, n, conf)
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


def loglog(x, n, conf=0.95):

    """
    Implements complimentary log-log interval for binary outcomes.

        Parameters:
            x (int or array): An integer or array of integers for the number of positive outcomes
            n (int or array): An integer or array of integers for the number of trials
            conf (float): The confidence level desired

        Returns:
            interval (dataframe):  A dataframe housing the risk estimate and upper/lower confidence bounds
    """

    _check_arguments(x, n, conf)
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


def wilson(x, n, conf=0.95):

    """
    Implements Wilson score interval for binary outcomes.

        Parameters:
            x (int or array): An integer or array of integers for the number of positive outcomes
            n (int or array): An integer or array of integers for the number of trials
            conf (float): The confidence level desired

        Returns:
            interval (dataframe):  A dataframe housing the risk estimate and upper/lower confidence bounds
    """

    _check_arguments(x, n, conf)
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


def exact(x, n, conf=0.95):

    """
    Implements exact confidence intervals for the binomial using results from the incomplete beta function.

        Parameters:
            x (int or array): An integer or array of integers for the number of positive outcomes
            n (int or array): An integer or array of integers for the number of trials
            conf (float): The confidence level desired

        Returns:
            interval (dataframe):  A dataframe housing the risk estimate and upper/lower confidence bounds
    """

    _check_arguments(x, n, conf)
    alpha = 1 - conf
    p = x / n
    lower = 1 - beta(a=n + 1 - x, b=x).ppf(1 - alpha / 2)
    upper = 1 - beta(a=n - x, b=x + 1).ppf(alpha / 2)
    interval = _interval(estimate=p, lower=lower, upper=upper, method="Exact")

    return interval
