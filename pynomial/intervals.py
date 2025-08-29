from dataclasses import dataclass
import warnings
import numpy as np
from scipy import stats
from enum import Enum


class ConfidenceIntervalType(str, Enum):
    """
    Enumeration of available confidence interval methods for binomial proportions.

    This enum defines the supported methods for computing confidence intervals,
    each with different statistical properties and use cases.

    Attributes:
        WALD: Wald interval based on normal approximation
        AGRESTI_COULL: Agresti-Coull interval with pseudo-observations
        ARCSINE: Arcsine transformation interval
        BAYESIAN_BETA: Bayesian interval using Beta prior
        CLOPPER_PEARSON: Exact (Clopper-Pearson) interval
        WILSON: Wilson (score) interval
        LOGIT: Logit transformation interval
        CLOGLOG: Complementary log-log transformation interval
    :meta private:
    """

    WALD = "wald"
    AGRESTI_COULL = "agresti_coull"
    ARCSINE = "arcsine"
    BAYESIAN_BETA = "bayesian_beta"
    CLOPPER_PEARSON = "clopper_pearson"
    WILSON = "wilson"
    LOGIT = "logit"
    CLOGLOG = "cloglog"


@dataclass
class ConfidenceInterval:
    """
    Result object containing confidence interval information.

    This dataclass holds all the information about a computed confidence interval
    including the method used, confidence level, point estimate, and bounds.

    Attributes:
        type (ConfidenceIntervalType): The method used to compute the interval
        confidence_level (float): The confidence level (e.g., 0.95 for 95%)
        point_estimate (float): The point estimate of the proportion
        lower (float): Lower bound of the confidence interval
        upper (float): Upper bound of the confidence interval

    :meta private:
    """

    type: ConfidenceIntervalType
    confidence_level: float
    point_estimate: float
    lower: float
    upper: float


def _z_score(confidence_level: float = 0.95) -> float:
    """
    Calculate the critical z-score for a given confidence level.

    Args:
        confidence_level (float): Confidence level between 0 and 1 (default: 0.95)

    Returns:
        float: The critical z-score corresponding to the confidence level

    Example:
        For 95% confidence level, returns approximately 1.96
    """
    return stats.norm.ppf(1 - (1 - confidence_level) / 2)


def _check_inputs(x, n, confidence_level: float = 0.95):
    """
    Validate input parameters for confidence interval functions.

    Performs comprehensive validation of input parameters including:
    - None value checks
    - Array broadcasting compatibility
    - Valid ranges for all parameters

    Args:
        x: Number of successes (scalar or array-like, 0 ≤ x ≤ n)
        n: Number of trials (scalar or array-like, n > 0)
        confidence_level (float): Confidence level between 0 and 1 (default: 0.95)

    Raises:
        ValueError: If any input parameter is invalid or if arrays are not broadcastable


    """
    if x is None or n is None:
        raise ValueError("x and n must not be None")

    x_arr = np.asarray(x)
    n_arr = np.asarray(n)
    confidence_level_arr = np.asarray(confidence_level)

    try:
        np.broadcast_shapes(x_arr.shape, n_arr.shape, confidence_level_arr.shape)
    except ValueError:
        raise ValueError(
            "x, n, and confidence_level must be broadcastable to the same shape"
        )

    if np.any(x_arr < 0):
        raise ValueError("x must be between 0 and n")
    if np.any(n_arr < 0):
        raise ValueError("n must be greater than 0")
    if np.any(x_arr > n_arr):
        raise ValueError("x must be between 0 and n")
    if np.any(confidence_level_arr < 0) or np.any(confidence_level_arr > 1):
        raise ValueError("confidence_level must be between 0 and 1")


def _highest_density_interval_beta(
    a: float, b: float, confidence_level: float, tol: float = 1e-8, max_iter: int = 1000
) -> tuple[float, float]:
    """
    Compute highest posterior density interval for Beta distribution.

    This function finds the shortest interval that contains the specified
    probability mass for a Beta distribution. The highest density interval
    minimizes the interval width while maintaining the desired coverage.

    Args:
        a (float): Alpha parameter (shape1) of the Beta distribution (a > 0)
        b (float): Beta parameter (shape2) of the Beta distribution (b > 0)
        confidence_level (float): Desired coverage probability (0 < confidence_level < 1)
        tol (float): Convergence tolerance for the iterative algorithm (default: 1e-8)
        max_iter (int): Maximum number of iterations (default: 1000)

    Returns:
        tuple[float, float]: Lower and upper bounds of the highest density interval


    """

    alpha = 1 - confidence_level

    lcl = stats.beta.ppf(alpha / 2, a, b)
    ucl = stats.beta.ppf(1 - alpha / 2, a, b)

    lcl_density = stats.beta.pdf(lcl, a, b)
    ucl_density = stats.beta.pdf(ucl, a, b)

    if a > 1 and b > 1:
        mode = (a - 1) / (a + b - 2)
    elif a <= 1 and b > 1:
        mode = 0.0
    elif a > 1 and b <= 1:
        mode = 1.0
    else:
        mode = 0.5

    target_density = max(lcl_density, ucl_density)
    y1 = 0.0
    y3 = target_density

    first_is_upper = lcl_density < ucl_density

    for i in range(max_iter):
        y2 = (y1 + y3) / 2

        try:
            if first_is_upper:
                lcl_new = _find_beta_density_root(a, b, y2, 0.0, mode, tol)
                ucl_new = _find_beta_density_root(a, b, y2, mode, 1.0, tol)
            else:
                lcl_new = _find_beta_density_root(a, b, y2, 0.0, mode, tol)
                ucl_new = _find_beta_density_root(a, b, y2, mode, 1.0, tol)

            coverage = stats.beta.cdf(ucl_new, a, b) - stats.beta.cdf(lcl_new, a, b)

            if abs(coverage - confidence_level) < tol:
                return lcl_new, ucl_new

            if coverage > confidence_level:
                y1 = y2
            else:
                y3 = y2

        except (ValueError, RuntimeError):
            break

    warnings.warn(
        "Failed to converge to highest density interval for Beta distribution; "
        "falling back to equal-tailed interval.",
        RuntimeWarning,
        stacklevel=2,
    )
    lcl = stats.beta.ppf(alpha / 2, a, b)
    ucl = stats.beta.ppf(1 - alpha / 2, a, b)
    return lcl, ucl


def _find_beta_density_root(
    a: float,
    b: float,
    target_density: float,
    x_lower: float,
    x_upper: float,
    tol: float,
) -> float:
    """
    Find the root where Beta distribution density equals target density.

    This helper function is used by the highest density interval algorithm
    to find points where the Beta probability density function equals a
    specified target value.

    Args:
        a (float): Alpha parameter of the Beta distribution
        b (float): Beta parameter of the Beta distribution
        target_density (float): Target density value to find roots for
        x_lower (float): Lower bound for root search
        x_upper (float): Upper bound for root search
        tol (float): Tolerance for root finding

    Returns:
        float: The x-value where pdf(x) ≈ target_density


    """
    from scipy.optimize import brentq

    def density_diff(x):
        return stats.beta.pdf(x, a, b) - target_density

    try:
        if density_diff(x_lower) * density_diff(x_upper) > 0:
            if abs(density_diff(x_lower)) < abs(density_diff(x_upper)):
                return x_lower
            else:
                return x_upper

        return brentq(density_diff, x_lower, x_upper, xtol=tol)
    except ValueError:
        return (x_lower + x_upper) / 2


def _continuity_corrected_wilson(
    x: int, n: int, confidence_level: float = 0.95
) -> ConfidenceInterval:
    """
    Wilson confidence interval with continuity correction.

    This implementation applies a continuity correction to the Wilson interval
    to better account for the discrete nature of binomial data. The correction
    generally improves coverage for small to moderate sample sizes.

    Args:
        x (int): Number of successes (0 ≤ x ≤ n)
        n (int): Number of trials (n > 0)
        confidence_level (float): Confidence level between 0 and 1 (default: 0.95)

    Returns:
        ConfidenceInterval: Object containing the confidence interval bounds and metadata

    :meta private:
    """
    _check_inputs(x, n, confidence_level)
    z_score = _z_score(confidence_level)
    pe = x / n

    z_squared = z_score**2
    two_n_pe = 2 * n * pe
    n_plus_z_squared = n + z_squared
    four_pe_minus_2 = 4 * pe - 2
    four_n_pe_one_minus_pe = 4 * n * pe * (1 - pe)

    sqrt_term_lower = z_squared - (1 / n) + four_n_pe_one_minus_pe + four_pe_minus_2
    sqrt_term_upper = z_squared - (1 / n) + four_n_pe_one_minus_pe - four_pe_minus_2

    sqrt_lower = z_score * np.sqrt(sqrt_term_lower)
    sqrt_upper = z_score * np.sqrt(sqrt_term_upper)

    lower_numerator = two_n_pe + z_squared - sqrt_lower - 1
    upper_numerator = two_n_pe + z_squared + sqrt_upper + 1

    lower = max(0, lower_numerator / (2 * n_plus_z_squared))
    upper = min(1, upper_numerator / (2 * n_plus_z_squared))

    return ConfidenceInterval(
        type=ConfidenceIntervalType.WILSON,
        confidence_level=confidence_level,
        point_estimate=pe,
        lower=lower,
        upper=upper,
    )


def _wilson_no_continuity(
    x: int, n: int, confidence_level: float = 0.95
) -> ConfidenceInterval:
    """
    Wilson confidence interval without continuity correction.

    This implementation computes the standard Wilson (score) interval without
    any continuity correction. It provides the basic Wilson interval formula
    derived from inverting the score test.

    Args:
        x (int): Number of successes (0 ≤ x ≤ n)
        n (int): Number of trials (n > 0)
        confidence_level (float): Confidence level between 0 and 1 (default: 0.95)

    Returns:
        ConfidenceInterval: Object containing the confidence interval bounds and metadata

    :meta private:
    """
    _check_inputs(x, n, confidence_level)
    z_score = _z_score(confidence_level)

    pe = x / n
    b = (z_score**2) / (2 * n) + pe
    discrim = z_score**2 / (4 * n) * (z_score**2 / n + 4 * pe * (1 - pe))
    denom = 1 + z_score**2 / n

    lower = (b - np.sqrt(discrim)) / denom
    upper = (b + np.sqrt(discrim)) / denom

    return ConfidenceInterval(
        type=ConfidenceIntervalType.WILSON,
        confidence_level=confidence_level,
        point_estimate=pe,
        lower=lower,
        upper=upper,
    )


def wald(x: int, n: int, confidence_level: float = 0.95) -> ConfidenceInterval:
    """
    Wald confidence interval for binomial proportions based off the central limit theorem.

    .. math::
        CI = \\hat{p} \\pm z_{1 - \\alpha/2} \\sqrt{\\frac{\\hat{p}(1-\\hat{p})}{n}}

    Args:
        x (int): Number of successes (0 ≤ x ≤ n)
        n (int): Number of trials (n > 0)
        confidence_level (float): Confidence level between 0 and 1 (default: 0.95)

    Returns:
        ConfidenceInterval: Object containing the confidence interval bounds and metadata
    """
    _check_inputs(x, n, confidence_level)
    z_score = _z_score(confidence_level)
    pe = x / n
    standard_error = np.sqrt(pe * (1 - pe) / n)
    lower = pe - z_score * standard_error
    upper = pe + z_score * standard_error

    return ConfidenceInterval(
        type=ConfidenceIntervalType.WALD,
        confidence_level=confidence_level,
        point_estimate=pe,
        lower=lower,
        upper=upper,
    )


def agresti_coull(x: int, n: int, confidence_level: float = 0.95) -> ConfidenceInterval:
    """
    Agresti-Coull confidence interval for binomial proportions from Agresti and Coull (1998) [1].

    .. math::
        \\tilde{x} &= x + \\frac{z^2}{2} \\\\
        \\tilde{n} &= n + z^2 \\\\
        \\tilde{p} &= \\frac{\\tilde{x}}{\\tilde{n}} \\\\
        CI &= \\tilde{p} \\pm z_{1 - \\alpha/2} \\sqrt{\\frac{\\tilde{p}(1-\\tilde{p})}{\\tilde{n}}}

    Args:
        x (int): Number of successes (0 ≤ x ≤ n)
        n (int): Number of trials (n > 0)
        confidence_level (float): Confidence level between 0 and 1 (default: 0.95)

    Returns:
        ConfidenceInterval: Object containing the confidence interval bounds and metadata

    References
    ----------
    .. [1] Agresti, Alan, and Brent A. Coull. "Approximate is better than "exact" for interval estimation of binomial proportions."
           The American Statistician 52.2 (1998): 119-126.
    """
    _check_inputs(x, n, confidence_level)
    z_score = _z_score(confidence_level)
    adjusted_x = x + z_score**2 / 2
    adjusted_n = n + z_score**2
    raw_pe = x / n
    pe = adjusted_x / adjusted_n
    standard_error = np.sqrt(pe * (1 - pe) / adjusted_n)
    lower = pe - z_score * standard_error
    upper = pe + z_score * standard_error

    return ConfidenceInterval(
        type=ConfidenceIntervalType.AGRESTI_COULL,
        confidence_level=confidence_level,
        point_estimate=raw_pe,
        lower=lower,
        upper=upper,
    )


def arcsine(x: int, n: int, confidence_level: float = 0.95) -> ConfidenceInterval:
    """
    Arcsine confidence interval for binomial proportions. The addition of 0.375 and 0.75 is an Anscombe correction [1].

    .. math::
        \\tilde{p} &= \\frac{x + 0.375}{n + 0.75} \\\\
        \\theta &= \\arcsin(\\sqrt{\\tilde{p}}) \\\\
        \\text{margin} &= \\frac{z_{1-\\alpha/2}}{2\\sqrt{n}} \\\\
        CI &= [\\sin^2(\\theta - \\text{margin}), \\sin^2(\\theta + \\text{margin})]

    Args:
        x (int): Number of successes (0 ≤ x ≤ n)
        n (int): Number of trials (n > 0)
        confidence_level (float): Confidence level between 0 and 1 (default: 0.95)

    Returns:
        ConfidenceInterval: Object containing the confidence interval bounds and metadata

    References
    ----------
    .. [1] Anscombe, Francis J. "The transformation of Poisson, binomial and negative-binomial data." Biometrika 35.3/4 (1948): 246-254.
    """
    _check_inputs(x, n, confidence_level)
    z_score = _z_score(confidence_level)

    pe = x / n

    # Anscombe correction
    p_tilde = (x + 0.375) / (n + 0.75)
    arcsine_p_tilde = np.arcsin(np.sqrt(p_tilde))

    margin = 0.5 * z_score / np.sqrt(n)
    lower = np.sin(arcsine_p_tilde - margin) ** 2
    upper = np.sin(arcsine_p_tilde + margin) ** 2

    return ConfidenceInterval(
        type=ConfidenceIntervalType.ARCSINE,
        confidence_level=confidence_level,
        point_estimate=pe,
        lower=lower,
        upper=upper,
    )


def bayesian_beta(
    x: int,
    n: int,
    prior_alpha: float = 0.5,
    prior_beta: float = 0.5,
    confidence_level: float = 0.95,
    interval_type: str = "central",
    tol: float = 1e-8,
    max_iter: int = 1000,
) -> ConfidenceInterval:
    """
    Bayesian confidence interval using Beta prior.  A conjugate beta prior is used so that the prior is

    .. math::
        \\Pr(p) = \\dfrac{p^{a}(1-p)^{b}}{B(a,b)}

    where :math:`B(a,b)` is the beta function.  Default values of 0.5 result in a Jeffreys prior.

    The posterior is then given by

    .. math::
        \\Pr(p|x) = \\dfrac{p^{x+a}(1-p)^{n-x+b}}{B(x+a,n-x+b)}


    Args:
        x (int): Number of successes (0 ≤ x ≤ n)
        n (int): Number of trials (n > 0)
        prior_alpha (float): Prior alpha parameter (shape1), default 0.5 (Jeffreys prior)
        prior_beta (float): Prior beta parameter (shape2), default 0.5 (Jeffreys prior)
        confidence_level (float): Confidence level between 0 and 1 (default: 0.95)
        interval_type (str): Either "central" (equal-tailed) or "highest" (highest density interval)
        tol (float): Tolerance for convergence (used for highest density intervals)
        max_iter (int): Maximum iterations for convergence

    Returns:
        ConfidenceInterval: Object containing the confidence interval bounds and metadata
    """
    _check_inputs(x, n, confidence_level)

    if prior_alpha <= 0 or prior_beta <= 0:
        raise ValueError("prior_alpha and prior_beta must be greater than 0")

    if interval_type not in ["central", "highest"]:
        raise ValueError("interval_type must be 'central' or 'highest'")

    a = x + prior_alpha
    b = n - x + prior_beta

    posterior_mean = a / (a + b)

    alpha = 1 - confidence_level

    if x == 0:
        lower = 0.0
        if interval_type == "central":
            upper = stats.beta.ppf(1 - alpha, a, b)
        else:
            upper = stats.beta.ppf(1 - alpha, a, b)
    elif x == n:
        upper = 1.0
        if interval_type == "central":
            lower = stats.beta.ppf(alpha, a, b)
        else:
            lower = stats.beta.ppf(alpha, a, b)
    else:
        if interval_type == "central":
            lower = stats.beta.ppf(alpha / 2, a, b)
            upper = stats.beta.ppf(1 - alpha / 2, a, b)
        else:
            lower, upper = _highest_density_interval_beta(
                a, b, confidence_level, tol, max_iter
            )

    return ConfidenceInterval(
        type=ConfidenceIntervalType.BAYESIAN_BETA,
        confidence_level=confidence_level,
        point_estimate=posterior_mean,
        lower=lower,
        upper=upper,
    )


def clopper_pearson(
    x: int, n: int, confidence_level: float = 0.95
) -> ConfidenceInterval:
    """
    Clopper-Pearson exact confidence interval for binomial proportions.

    .. math::
        \\text{Lower bound} &= \\text{Beta}_{\\alpha/2}(x, n-x+1) \\\\
        \\text{Upper bound} &= \\text{Beta}_{1-\\alpha/2}(x+1, n-x)

    Args:
        x (int): Number of successes (0 ≤ x ≤ n)
        n (int): Number of trials (n > 0)
        confidence_level (float): Confidence level between 0 and 1 (default: 0.95)

    Returns:
        ConfidenceInterval: Object containing the confidence interval bounds and metadata

    Special Cases:
        - When x=0: lower bound = 0
        - When x=n: upper bound = 1
    """
    _check_inputs(x, n, confidence_level)
    pe = x / n
    alpha = 1 - confidence_level

    if x == 0:
        lower = 0.0
        upper = stats.beta.ppf(1 - alpha / 2, x + 1, n - x)
    elif x == n:
        lower = stats.beta.ppf(alpha / 2, x, n - x + 1)
        upper = 1.0
    else:
        lower = stats.beta.ppf(alpha / 2, x, n - x + 1)
        upper = stats.beta.ppf(1 - alpha / 2, x + 1, n - x)

    return ConfidenceInterval(
        type=ConfidenceIntervalType.CLOPPER_PEARSON,
        confidence_level=confidence_level,
        point_estimate=pe,
        lower=lower,
        upper=upper,
    )


def wilson(
    x: int, n: int, correct_continuity: bool = True, confidence_level: float = 0.95
) -> ConfidenceInterval:
    """
        Wilson confidence interval for binomial proportions.

    The Wilson (or score) interval is obtained by inverting the score test 
    for the binomial proportion. Unlike the Wald interval, it avoids 
    poor coverage when n is small or p is near 0 or 1. 

    The interval arises from solving the quadratic inequality [1]:

    .. math::
        n (\hat{p} - p)^2 \leq z_{1 - \\alpha/2}^2 \, p (1 - p)

    where :math:`\hat{p} = x/n` is the sample proportion and 
    :math:`z_{\\alpha/2}` is the standard normal quantile.

    Solving for :math:`p` gives the Wilson bounds:

    .. math::
        \\text{Lower}, \\text{Upper} = 
        \\frac{\\hat{p} + \\tfrac{z^2}{2n} \\pm 
        z \\sqrt{ \\tfrac{\\hat{p}(1 - \\hat{p})}{n} + \\tfrac{z^2}{4n^2} } }
        {1 + \\tfrac{z^2}{n}}

    Optionally, a continuity correction can be applied to better account 
    for the discreteness of the binomial distribution.

    Args:
        x (int): Number of successes (0 ≤ x ≤ n).
        n (int): Number of trials (n > 0).
        correct_continuity (bool): Whether to apply continuity correction 
            (default: True).
        confidence_level (float): Confidence level between 0 and 1 
            (default: 0.95).

    Returns:
        ConfidenceInterval: Object containing the confidence interval bounds and metadata

    References
    ----------
    .. [1] Lachin, John M. Biostatistical methods: the assessment of relative risks. John Wiley & Sons, 2014.
    """
    if correct_continuity:
        return _continuity_corrected_wilson(x, n, confidence_level)
    else:
        return _wilson_no_continuity(x, n, confidence_level)


def logit(x: int, n: int, confidence_level: float = 0.95) -> ConfidenceInterval:
    """
    Logit transformation confidence interval for binomial proportions [1].

    .. math::
        \\text{logit}(\\hat{p}) &= \\ln\\left(\\frac{\\hat{p}}{1-\\hat{p}}\\right) \\\\
        SE_{\\text{logit}} &= \\sqrt{\\frac{1}{n\\hat{p}(1-\\hat{p})}} \\\\
        \\text{logit}(CI) &= \\text{logit}(\\hat{p}) \\pm z_{1-\\alpha/2} \\cdot SE_{\\text{logit}} \\\\
        CI &= \\left[\\frac{e^{L}}{1+e^{L}}, \\frac{e^{U}}{1+e^{U}}\\right]

    Args:
        x (int): Number of successes (0 ≤ x ≤ n)
        n (int): Number of trials (n > 0)
        confidence_level (float): Confidence level between 0 and 1 (default: 0.95)

    Returns:
        ConfidenceInterval: Object containing the confidence interval bounds and metadata

    Special Cases:
        - When x=0: lower = 0, upper calculated using beta distribution
        - When x=n: upper = 1, lower calculated using beta distribution

    References
    ----------
    .. [1] Lachin, John M. Biostatistical methods: the assessment of relative risks. John Wiley & Sons, 2014.
    """
    _check_inputs(x, n, confidence_level)
    z_score = _z_score(confidence_level)
    pe = x / n

    if x == 0:
        lower = 0.0
        upper = stats.beta.ppf(1 - (1 - confidence_level) / 2, x + 1, n - x)

    elif x == n:
        lower = stats.beta.ppf((1 - confidence_level) / 2, x, n - x + 1)
        upper = 1.0

    else:
        logit_pe = np.log(pe / (1 - pe))
        standard_error = np.sqrt(1 / (n * pe * (1 - pe)))

        logit_lower = logit_pe - z_score * standard_error
        logit_upper = logit_pe + z_score * standard_error

        lower = np.exp(logit_lower) / (1 + np.exp(logit_lower))
        upper = np.exp(logit_upper) / (1 + np.exp(logit_upper))

    return ConfidenceInterval(
        type=ConfidenceIntervalType.LOGIT,
        confidence_level=confidence_level,
        point_estimate=pe,
        lower=lower,
        upper=upper,
    )


def cloglog(x: int, n: int, confidence_level: float = 0.95) -> ConfidenceInterval:
    """
    Complementary log-log transformation confidence interval for binomial proportions [1].

    .. math::
        \\text{cloglog}(\\hat{p}) &= \\ln(-\\ln(1-\\hat{p})) \\\\
        \\mu &= \\ln(\\hat{p}) \\\\
        \\text{Var}[\\text{cloglog}(\\hat{p})] &= \\frac{1-\\hat{p}}{n\\hat{p}\\mu^2} \\\\
        \\text{cloglog}(CI) &= \\text{cloglog}(\\hat{p}) \\pm z_{1-\\alpha/2} \\sqrt{\\text{Var}[\\text{cloglog}(\\hat{p})]} \\\\
        CI &= [\\exp(-\\exp(U)), \\exp(-\\exp(L))]

    Args:
        x (int): Number of successes (0 ≤ x ≤ n)
        n (int): Number of trials (n > 0)
        confidence_level (float): Confidence level between 0 and 1 (default: 0.95)

    Returns:
        ConfidenceInterval: Object containing the confidence interval bounds and metadata

    Special Cases:
        - When x=0: lower = 0, upper = :math:`1 - (\\alpha/2)^{1/n}`
        - When x=n: lower = :math:`(\\alpha/2)^{1/n}`, upper = 1

    References
    ----------
    .. [1] Lachin, John M. Biostatistical methods: the assessment of relative risks. John Wiley & Sons, 2014.
    """
    _check_inputs(x, n, confidence_level)
    z_score = _z_score(confidence_level)
    pe = x / n
    alpha = 1 - confidence_level
    alpha2 = 0.5 * alpha
    
    if x == 0:
        lower = 0.0
        upper = 1 - alpha2 ** (1 / n)
    elif x == n:
        lower = alpha2 ** (1 / n)
        upper = 1.0
    else:
        log_mu = np.log(-np.log(pe))
        mu = np.log(pe)
        var_cloglog = (1 - pe) / (n * pe * mu**2)
        sd = np.sqrt(var_cloglog)
        
        lcl_loglog = log_mu + z_score * sd
        ucl_loglog = log_mu - z_score * sd
        
        lower = np.exp(-np.exp(lcl_loglog))
        upper = np.exp(-np.exp(ucl_loglog))

    return ConfidenceInterval(
        type=ConfidenceIntervalType.CLOGLOG,
        confidence_level=confidence_level,
        point_estimate=pe,
        lower=lower,
        upper=upper,
    )
