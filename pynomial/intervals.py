from dataclasses import dataclass
import numpy as np
from scipy import stats
from enum import Enum

class ConfidenceIntervalType(str, Enum):
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
    type: ConfidenceIntervalType
    confidence_level: float
    point_estimate: float
    lower: float
    upper: float


def _z_score(confidence_level: float = 0.95) -> float:
    return stats.norm.ppf(1 - (1 - confidence_level) / 2)


def _check_inputs(x, n, confidence_level: float = 0.95):
    # Check for None
    if x is None or n is None:
        raise ValueError("x and n must not be None")

    # Convert to numpy arrays for shape checking
    x_arr = np.asarray(x)
    n_arr = np.asarray(n)
    confidence_level_arr = np.asarray(confidence_level)

    try:
        np.broadcast_shapes(x_arr.shape, n_arr.shape, confidence_level_arr.shape)
    except ValueError:
        raise ValueError("x, n, and confidence_level must be broadcastable to the same shape")

    # Check value constraints
    if np.any(x_arr < 0):
        raise ValueError("x must be between 0 and n")
    if np.any(n_arr < 0):
        raise ValueError("n must be greater than 0")
    if np.any(x_arr > n_arr):
        raise ValueError("x must be between 0 and n")
    if np.any(confidence_level_arr < 0) or np.any(confidence_level_arr > 1):
        raise ValueError("confidence_level must be between 0 and 1")


def wald(x: int, n: int, confidence_level: float = 0.95) -> ConfidenceInterval:
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
    _check_inputs(x, n, confidence_level)
    z_score = _z_score(confidence_level)
    pe = x / n
    arcsine_pe = np.arcsin(np.sqrt(pe))
    lower = np.sin(arcsine_pe - z_score / np.sqrt(4 * n)) ** 2
    upper = np.sin(arcsine_pe + z_score / np.sqrt(4 * n)) ** 2

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
    prior_alpha: float = 1,
    prior_beta: float = 1,
    confidence_level: float = 0.95,
) -> ConfidenceInterval:
    _check_inputs(x, n, confidence_level)
    posterior = stats.beta(prior_alpha + x, prior_beta + n - x)
    posterior_mean = (prior_alpha + x) / (prior_alpha + prior_beta + n)
    lower = posterior.ppf((1 - confidence_level) / 2)
    upper = posterior.ppf(1 - (1 - confidence_level) / 2)

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
    _check_inputs(x, n, confidence_level)
    pe = x / n
    lower = stats.beta.ppf((1 - confidence_level) / 2, x, n - x + 1)
    upper = stats.beta.ppf(1 - (1 - confidence_level) / 2, x + 1, n - x)

    return ConfidenceInterval(
        type=ConfidenceIntervalType.CLOPPER_PEARSON,
        confidence_level=confidence_level,
        point_estimate=pe,
        lower=lower,
        upper=upper,
    )


def _continuity_corrected_wilson(x: int, n: int, confidence_level: float = 0.95) -> ConfidenceInterval:
    """
    Wilson confidence interval with continuity correction.

    This implementation uses the continuity-corrected Wilson interval formulas:
    w_cc^- = max{0, (2np̂ + z_α² - [z_α√(z_α² - 1/n + 4np̂(1-p̂) + (4p̂-2)) + 1]) / (2(n + z_α²))}
    w_cc^+ = min{1, (2np̂ + z_α² + [z_α√(z_α² - 1/n + 4np̂(1-p̂) - (4p̂-2)) + 1]) / (2(n + z_α²))}
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

def _wilson_no_continuity(x: int, n: int, confidence_level: float = 0.95) -> ConfidenceInterval:
    _check_inputs(x, n, confidence_level)
    z_score = _z_score(confidence_level)

    pe = x / n
    b =  (z_score**2) / (2 * n)  + pe
    discrim = z_score ** 2 / (4 * n) * (z_score ** 2 / n + 4 * pe * (1 - pe))
    denom = 1 + z_score ** 2 / n

    lower = (b - np.sqrt(discrim)) / denom
    upper = (b + np.sqrt(discrim)) / denom

    return ConfidenceInterval(
        type=ConfidenceIntervalType.WILSON,
        confidence_level=confidence_level,
        point_estimate=pe,
        lower=lower,
        upper=upper,
    )

def wilson(x: int, n: int, correct_continuity: bool = True, confidence_level: float = 0.95) -> ConfidenceInterval:
    if correct_continuity:
        return _continuity_corrected_wilson(x, n, confidence_level)
    else:
        return _wilson_no_continuity(x, n, confidence_level)

def logit(x: int, n: int, confidence_level: float = 0.95) -> ConfidenceInterval:
    _check_inputs(x, n, confidence_level)
    z_score = _z_score(confidence_level)
    pe = x / n
    standard_error = np.sqrt(1 / (n * pe * (1 - pe)))
    logit_lower = pe - z_score * standard_error
    logit_upper = pe + z_score * standard_error
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
    _check_inputs(x, n, confidence_level)
    z_score = _z_score(confidence_level)
    pe = x / n
    alpha = 1 - confidence_level
    alpha2 = 0.5 * alpha
    
    # Handle edge cases like R implementation
    if x == 0:
        lower = 0.0
        upper = 1 - alpha2**(1/n)
    elif x == n:
        lower = alpha2**(1/n)
        upper = 1.0
    else:
        # Standard cloglog calculation for interior cases
        log_mu = np.log(-np.log(pe))
        # Variance formula from R's var.cloglog: (1-p) / (n * p * (log(p))^2)
        # Note: this uses log(p), not log(-log(p))
        mu = np.log(pe)
        var_cloglog = (1 - pe) / (n * pe * mu**2)
        sd = np.sqrt(var_cloglog)
        
        # Calculate confidence interval on log(-log(p)) scale
        lcl_loglog = log_mu + z_score * sd
        ucl_loglog = log_mu - z_score * sd
        
        # Transform back to probability scale
        lower = np.exp(-np.exp(lcl_loglog))
        upper = np.exp(-np.exp(ucl_loglog))

    return ConfidenceInterval(
        type=ConfidenceIntervalType.CLOGLOG,
        confidence_level=confidence_level,
        point_estimate=pe,
        lower=lower,
        upper=upper,
    )

