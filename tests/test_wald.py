from pytest import mark, fixture
import numpy as np
from pynomial.intervals import wald

TOLERANCE = 1e-6

@fixture
def wald_output():
    """
    Test data for the Wald confidence interval.
    Generated from R's binom library using the following code:

    #> binom::binom.asymp(0:10, 10, conf.level = 0.95)
    #> binom::binom.asymp(0:10, 10, conf.level = 0.8)
    """
    return [
        # 95% confidence level cases
        (0, 10, 0.95, 0.0, 0.00000000, 0.0000000),
        (1, 10, 0.95, 0.1, -0.08593851, 0.2859385),
        (2, 10, 0.95, 0.2, -0.04791801, 0.4479180),
        (3, 10, 0.95, 0.3, 0.01597423, 0.5840258),
        (4, 10, 0.95, 0.4, 0.09636369, 0.7036363),
        (5, 10, 0.95, 0.5, 0.19010248, 0.8098975),
        (6, 10, 0.95, 0.6, 0.29636369, 0.9036363),
        (7, 10, 0.95, 0.7, 0.41597423, 0.9840258),
        (8, 10, 0.95, 0.8, 0.55208199, 1.0479180),
        (9, 10, 0.95, 0.9, 0.71406149, 1.0859385),
        (10, 10, 0.95, 1.0, 1.00000000, 1.0000000),
        # 80% confidence level cases
        (0, 10, 0.8, 0.0, 0.00000000, 0.0000000),
        (1, 10, 0.8, 0.1, -0.02157866, 0.2215787),
        (2, 10, 0.8, 0.2, 0.03789512, 0.3621049),
        (3, 10, 0.8, 0.3, 0.11428553, 0.4857145),
        (4, 10, 0.8, 0.4, 0.20146289, 0.5985371),
        (5, 10, 0.8, 0.5, 0.29736891, 0.7026311),
        (6, 10, 0.8, 0.6, 0.40146289, 0.7985371),
        (7, 10, 0.8, 0.7, 0.51428553, 0.8857145),
        (8, 10, 0.8, 0.8, 0.63789512, 0.9621049),
        (9, 10, 0.8, 0.9, 0.77842134, 1.0215787),
        (10, 10, 0.8, 1.0, 1.00000000, 1.0000000),
    ]


@mark.wald
class TestWaldInterval:
    """Test suite for the Wald confidence interval."""
    
    def test_asymptotic_calculations(self, wald_output):
        """Test Wald interval calculations against known reference values."""
        for x, n, conf_level, expected_mean, expected_lower, expected_upper in wald_output:
            result = wald(x, n, confidence_level=conf_level)
            
            # Test point estimate (mean)
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
        result = wald(5, 10)
        assert result.point_estimate == 0.5
        assert result.lower < result.point_estimate < result.upper
        assert result.confidence_level == 0.95

    def test_zero_successes(self):
        """Test edge case with zero successes."""
        result = wald(0, 10)
        assert result.point_estimate == 0.0
        assert result.lower == 0.0
        assert result.upper == 0.0

    def test_zero_failures(self):
        """Test edge case with zero failures (all successes)."""
        result = wald(10, 10)
        assert result.point_estimate == 1.0
        assert result.lower == 1.0
        assert result.upper == 1.0
