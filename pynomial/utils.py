import numpy as np
from scipy.special import expit, logit
from scipy.optimize import newton
from scipy.stats import binom, chi2


def _check_args(x, n, conf):

    x = np.asarray(x)
    n = np.asarray(n)
    conf = np.asarray(conf)

    # Check x contains ints
    if np.any(np.mod(x, 1) != 0):
        raise TypeError("x must be an integer or an array of integers")

    # Check n contains only ints
    if np.any(np.mod(n, 1) != 0):
        raise TypeError("n must be an integer or an array of integers")

    # Check x is a non negative int which does not exceed n
    if np.any(x < 0) or np.any(n < x):
        raise ValueError(f"x must be a non-negative integer less than or equal to n")

    # Check no trials are 0 or negative
    if np.any(n <= 0):
        raise ValueError(f"n must be a positive integer")

    # Check conf is realistic
    if np.any(conf <= 0) or np.any(1 <= conf):
        raise ValueError("conf must be a number between 0 and 1")

    # Ensure that x, n, and conf are arrays of the same dimension
    lens = np.unique((x.size, n.size, conf.size))
    if len(lens) > 2:
        raise ValueError("If passing arrays as arguments, arrays must be same length")

    max_len = max(x.size, n.size, conf.size)

    if x.size < max_len:
        x = np.ones(max_len) * x
    if n.size < max_len:
        n = np.ones(max_len) * n
    if conf.size < max_len:
        conf = np.ones(max_len) * conf

    return x, n, conf


def _lrt_rootfinding(x, n, conf, *args, **kwargs):

    """
    Inversion of the likelihood ratio test statistic to construct profile likelihood intervals
    requires rootfinding techniques.  This function performs the neccesary rootfinding for
    pynomial.intervals.lrt
    """

    # x, n, conf are all arrays of the same size when passed to this function
    # For each triplet, we will estimate the upper and lower endpoints on the logit scale
    # for numerical stability.
    data = np.c_[x, n, conf]
    logit_lower = np.zeros(len(data))
    logit_upper = np.zeros(len(data))

    for i, (xi, ni, confi) in enumerate(data):
        p = xi / ni
        alpha = 1 - confi
        X = chi2(df=1).ppf(1 - alpha)

        log_lik_est = binom(n=ni, p=p).logpmf(xi)

        # For numerically stable reasons, we pass logit p to the log likelihood and
        # Perform the root finding in the log odds space
        # Then, transform back to probability space
        lrt_stat = (
            lambda logit_p: -2
            * (binom(n=ni, p=expit(logit_p)).logpmf(xi) - log_lik_est)
            - X
        )

        logit_lower[i] = newton(lrt_stat, x0=logit(p) - 2, **kwargs)
        logit_upper[i] = newton(lrt_stat, x0=logit(p) + 2, **kwargs)
    lower = expit(logit_lower)
    upper = expit(logit_upper)

    return lower, upper
