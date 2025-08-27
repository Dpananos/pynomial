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


def _highest_density_interval_beta(a: float, b: float, confidence_level: float, tol: float = 1e-8, max_iter: int = 1000) -> tuple[float, float]:
    """
    Compute highest posterior density interval for Beta distribution.
    
    This implements the algorithm from R's binom package C code.
    """
    alpha = 1 - confidence_level
    
    # Initial bounds using equal-tailed interval
    lcl = stats.beta.ppf(alpha / 2, a, b)
    ucl = stats.beta.ppf(1 - alpha / 2, a, b)
    
    # Get densities at initial bounds
    lcl_density = stats.beta.pdf(lcl, a, b)
    ucl_density = stats.beta.pdf(ucl, a, b)
    
    # Find the mode
    if a > 1 and b > 1:
        mode = (a - 1) / (a + b - 2)
    elif a <= 1 and b > 1:
        mode = 0.0
    elif a > 1 and b <= 1:
        mode = 1.0
    else:
        mode = 0.5  # a <= 1 and b <= 1
    
    # Start with the higher density
    target_density = max(lcl_density, ucl_density)
    y1 = 0.0
    y3 = target_density
    
    # Determine search direction based on which bound has higher density
    first_is_upper = lcl_density < ucl_density
    
    # Binary search for the density level that gives the right coverage
    for i in range(max_iter):
        y2 = (y1 + y3) / 2
        
        # Find interval bounds for this density level
        try:
            # Find where beta density equals y2
            if first_is_upper:
                # Search for lower bound
                lcl_new = _find_beta_density_root(a, b, y2, 0.0, mode, tol)
                ucl_new = _find_beta_density_root(a, b, y2, mode, 1.0, tol)
            else:
                # Search for upper bound  
                lcl_new = _find_beta_density_root(a, b, y2, 0.0, mode, tol)
                ucl_new = _find_beta_density_root(a, b, y2, mode, 1.0, tol)
            
            # Calculate coverage
            coverage = stats.beta.cdf(ucl_new, a, b) - stats.beta.cdf(lcl_new, a, b)
            
            if abs(coverage - confidence_level) < tol:
                return lcl_new, ucl_new
                
            if coverage > confidence_level:
                y1 = y2  # Need higher density (smaller interval)
            else:
                y3 = y2  # Need lower density (larger interval)
                
        except (ValueError, RuntimeError):
            # If root finding fails, fall back to equal-tailed
            break
    
    # Fallback to equal-tailed interval if HDI computation fails
    lcl = stats.beta.ppf(alpha / 2, a, b)
    ucl = stats.beta.ppf(1 - alpha / 2, a, b)
    return lcl, ucl


def _find_beta_density_root(a: float, b: float, target_density: float, x_lower: float, x_upper: float, tol: float) -> float:
    """Find x such that beta.pdf(x, a, b) = target_density using Brent's method."""
    from scipy.optimize import brentq
    
    def density_diff(x):
        return stats.beta.pdf(x, a, b) - target_density
    
    try:
        # Check if root exists in interval
        if density_diff(x_lower) * density_diff(x_upper) > 0:
            # No sign change, return the endpoint with density closest to target
            if abs(density_diff(x_lower)) < abs(density_diff(x_upper)):
                return x_lower
            else:
                return x_upper
        
        return brentq(density_diff, x_lower, x_upper, xtol=tol)
    except ValueError:
        # Return midpoint if optimization fails
        return (x_lower + x_upper) / 2


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
    prior_alpha: float = 0.5,
    prior_beta: float = 0.5,
    confidence_level: float = 0.95,
    interval_type: str = "central",
    tol: float = 1e-8,
    max_iter: int = 1000,
) -> ConfidenceInterval:
    """
    Bayesian confidence interval using Beta prior.
    
    Based on R's binom.bayes function from the binom package.
    
    Args:
        x: Number of successes
        n: Number of trials
        prior_alpha: Prior alpha parameter (shape1), default 0.5 (Jeffreys prior)
        prior_beta: Prior beta parameter (shape2), default 0.5 (Jeffreys prior)
        confidence_level: Confidence level
        interval_type: Either "central" (equal-tailed) or "highest" (highest density interval)
        tol: Tolerance for convergence (used for highest density intervals)
        max_iter: Maximum iterations for convergence
    """
    _check_inputs(x, n, confidence_level)
    
    if prior_alpha <= 0 or prior_beta <= 0:
        raise ValueError("prior_alpha and prior_beta must be greater than 0")
        
    if interval_type not in ["central", "highest"]:
        raise ValueError("interval_type must be 'central' or 'highest'")
    
    # Posterior parameters
    a = x + prior_alpha
    b = n - x + prior_beta
    
    # Posterior mean
    posterior_mean = a / (a + b)
    
    # Calculate confidence interval
    alpha = 1 - confidence_level
    
    # Handle edge cases following R's binom.bayes implementation
    if x == 0:
        lower = 0.0
        if interval_type == "central":
            upper = stats.beta.ppf(1 - alpha, a, b)
        else:  # highest density
            upper = stats.beta.ppf(1 - alpha, a, b)
    elif x == n:
        upper = 1.0
        if interval_type == "central":
            lower = stats.beta.ppf(alpha, a, b)
        else:  # highest density
            lower = stats.beta.ppf(alpha, a, b)
    else:
        if interval_type == "central":
            # Equal-tailed interval
            lower = stats.beta.ppf(alpha / 2, a, b)
            upper = stats.beta.ppf(1 - alpha / 2, a, b)
        else:
            # Highest posterior density interval
            lower, upper = _highest_density_interval_beta(a, b, confidence_level, tol, max_iter)
    
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
    
    This method provides exact coverage by using the beta distribution relationship
    to the binomial distribution. The confidence interval is:
    
    lower = Beta((1-α)/2; x, n-x+1)
    upper = Beta(1-(1-α)/2; x+1, n-x)
    
    Special cases:
    - When x=0: lower = 0
    - When x=n: upper = 1
    """
    _check_inputs(x, n, confidence_level)
    pe = x / n
    alpha = 1 - confidence_level
    
    # Handle edge cases
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
    """
    Logit transformation confidence interval for binomial proportions.
    
    This method uses the logit transformation:
    1. Transform p to logit scale: logit(p) = ln(p/(1-p))
    2. Calculate CI on logit scale using normal approximation  
    3. Transform back to probability scale using inverse logit
    
    Special cases:
    - When x=0: lower = 0, upper calculated using continuity correction
    - When x=n: upper = 1, lower calculated using continuity correction
    """
    _check_inputs(x, n, confidence_level)
    z_score = _z_score(confidence_level)
    pe = x / n
    
    # Handle edge cases like R's implementation
    # R's binom.logit falls back to exact (Clopper-Pearson) for edge cases
    if x == 0:
        lower = 0.0
        # Use exact upper bound like Clopper-Pearson
        upper = stats.beta.ppf(1 - (1 - confidence_level) / 2, x + 1, n - x)
        
    elif x == n:
        # Use exact lower bound like Clopper-Pearson  
        lower = stats.beta.ppf((1 - confidence_level) / 2, x, n - x + 1)
        upper = 1.0
        
    else:
        # Standard logit transformation
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

