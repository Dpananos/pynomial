from pytest import mark, fixture
import numpy as np
from pynomial.intervals import bayesian_beta

TOLERANCE = 1e-6

@fixture
def bayesian_beta_highest_output():
    """
    Test data for the Bayesian beta confidence interval with highest density intervals.
    Generated from R's binom library using:
    
    binom::binom.bayes(0:10, 10, conf.level = 0.95, type='highest')
    
    Note: Using prior_shape1=0.5, prior_shape2=0.5 (Jeffreys prior)
    """
    return [
        # (x, n, conf_level, expected_mean, expected_lower, expected_upper)
        (0, 10, 0.95, 0.04545455, 0.0000000000, 0.1707731),
        (1, 10, 0.95, 0.13636364, 0.0003602864, 0.3308030),
        (2, 10, 0.95, 0.22727273, 0.0234655042, 0.4618984),
        (3, 10, 0.95, 0.31818182, 0.0745442290, 0.5794516),
        (4, 10, 0.95, 0.40909091, 0.1425673496, 0.6838697),
        (5, 10, 0.95, 0.50000000, 0.2235286703, 0.7764713),
        (6, 10, 0.95, 0.59090909, 0.3161302843, 0.8574327),
        (7, 10, 0.95, 0.68181818, 0.4205484202, 0.9254558),
        (8, 10, 0.95, 0.77272727, 0.5381015872, 0.9765345),
        (9, 10, 0.95, 0.86363636, 0.6691970292, 0.9996397),
        (10, 10, 0.95, 0.95454545, 0.8292268918, 1.0000000),
    ]

@fixture
def bayesian_beta_central_output():
    """
    Test data for the Bayesian beta confidence interval with central (equal-tailed) intervals.
    Generated from R's binom library using:
    
    binom::binom.bayes(0:10, 10, conf.level = 0.8, type='central')
    
    Note: Using prior_shape1=0.5, prior_shape2=0.5 (Jeffreys prior)
    """
    return [
        # (x, n, conf_level, expected_mean, expected_lower, expected_upper)
        (0, 10, 0.8, 0.04545455, 0.00000000, 0.0769566),
        (1, 10, 0.8, 0.13636364, 0.02954113, 0.2745618),
        (2, 10, 0.8, 0.22727273, 0.08361516, 0.3948296),
        (3, 10, 0.8, 0.31818182, 0.15059123, 0.5017811),
        (4, 10, 0.8, 0.40909091, 0.22651679, 0.5996844),
        (5, 10, 0.8, 0.50000000, 0.30989203, 0.6901080),
        (6, 10, 0.8, 0.59090909, 0.40031562, 0.7734832),
        (7, 10, 0.8, 0.68181818, 0.49821888, 0.8494088),
        (8, 10, 0.8, 0.77272727, 0.60517036, 0.9163848),
        (9, 10, 0.8, 0.86363636, 0.72543825, 0.9704589),
        (10, 10, 0.8, 0.95454545, 0.92304340, 1.0000000),
    ]


@mark.bayesian_beta
class TestBayesianBetaInterval:
    """Test suite for the Bayesian beta confidence interval."""
    
    def test_highest_density_interval_calculations(self, bayesian_beta_highest_output):
        """Test Bayesian beta HDI calculations against known R reference values."""
        for x, n, conf_level, expected_mean, expected_lower, expected_upper in bayesian_beta_highest_output:
            result = bayesian_beta(x, n, confidence_level=conf_level, interval_type='highest')
            
            # Test point estimate (posterior mean)
            assert np.isclose(result.point_estimate, expected_mean, rtol=TOLERANCE), \
                f"Point estimate mismatch for x={x}, n={n}, conf_level={conf_level}: got {result.point_estimate}, expected {expected_mean}"
            
            # Test lower bound - use more lenient tolerance for HDI as it's iterative
            assert np.isclose(result.lower, expected_lower, rtol=1e-3), \
                f"Lower bound mismatch for x={x}, n={n}, conf_level={conf_level}: got {result.lower}, expected {expected_lower}"
            
            # Test upper bound
            assert np.isclose(result.upper, expected_upper, rtol=1e-3), \
                f"Upper bound mismatch for x={x}, n={n}, conf_level={conf_level}: got {result.upper}, expected {expected_upper}"
    
    def test_central_interval_calculations(self, bayesian_beta_central_output):
        """Test Bayesian beta central interval calculations against known R reference values."""
        for x, n, conf_level, expected_mean, expected_lower, expected_upper in bayesian_beta_central_output:
            result = bayesian_beta(x, n, confidence_level=conf_level, interval_type='central')
            
            # Test point estimate (posterior mean)
            assert np.isclose(result.point_estimate, expected_mean, rtol=TOLERANCE), \
                f"Point estimate mismatch for x={x}, n={n}, conf_level={conf_level}: got {result.point_estimate}, expected {expected_mean}"
            
            # Test lower bound
            assert np.isclose(result.lower, expected_lower, rtol=TOLERANCE), \
                f"Lower bound mismatch for x={x}, n={n}, conf_level={conf_level}: got {result.lower}, expected {expected_lower}"
            
            # Test upper bound
            assert np.isclose(result.upper, expected_upper, rtol=TOLERANCE), \
                f"Upper bound mismatch for x={x}, n={n}, conf_level={conf_level}: got {result.upper}, expected {expected_upper}"

    def test_calculation(self):
        """Basic calculation test."""
        result = bayesian_beta(5, 10)
        assert result.lower < result.point_estimate < result.upper
        assert result.confidence_level == 0.95
        # With Jeffreys prior (0.5, 0.5), posterior mean = (x + 0.5)/(n + 1) = 5.5/11
        expected_mean = 5.5 / 11
        assert np.isclose(result.point_estimate, expected_mean, rtol=TOLERANCE)

    def test_interval_types(self):
        """Test that both interval types work and HDI is generally narrower."""
        central_result = bayesian_beta(5, 10, interval_type='central')
        hdi_result = bayesian_beta(5, 10, interval_type='highest')
        
        # Both should have same point estimate
        assert np.isclose(central_result.point_estimate, hdi_result.point_estimate, rtol=TOLERANCE)
        
        # HDI should generally be narrower (though not always for symmetric cases)
        central_width = central_result.upper - central_result.lower
        hdi_width = hdi_result.upper - hdi_result.lower
        # For this symmetric case, they should be very similar
        assert abs(central_width - hdi_width) < 0.1

    def test_edge_cases(self):
        """Test edge cases."""
        # x = 0 case
        result_zero = bayesian_beta(0, 10)
        assert result_zero.lower == 0.0
        assert result_zero.upper > 0.0
        
        # x = n case
        result_all = bayesian_beta(10, 10)
        assert result_all.lower < 1.0
        assert result_all.upper == 1.0

    def test_error_conditions(self):
        """Test error conditions."""
        from pytest import raises
        
        # Invalid prior parameters
        with raises(ValueError, match="prior_alpha and prior_beta must be greater than 0"):
            bayesian_beta(5, 10, prior_alpha=0.0)
        
        with raises(ValueError, match="prior_alpha and prior_beta must be greater than 0"):
            bayesian_beta(5, 10, prior_beta=-1.0)
        
        # Invalid interval type
        with raises(ValueError, match="interval_type must be 'central' or 'highest'"):
            bayesian_beta(5, 10, interval_type='invalid')

