from pytest import mark, fixture
import numpy as np
from pynomial.intervals import cloglog

TOLERANCE = 1e-6

@fixture
def cloglog_output():
    """
    Test data for the cloglog confidence interval.
    Generated from R's binom library using the following code:

    #> binom::binom.cloglog(0:10, 10, conf.level = 0.95)
    #> binom::binom.cloglog(0:10, 10, conf.level = 0.8)
    """
    return [
        # 95% confidence level cases
        (0, 10, 0.95, 0.0, 0.000000000, 0.3084971),
        (1, 10, 0.95, 0.1, 0.005723456, 0.3581275),
        (2, 10, 0.95, 0.2, 0.030909024, 0.4747147),
        (3, 10, 0.95, 0.3, 0.071134492, 0.5778673),
        (4, 10, 0.95, 0.4, 0.122693166, 0.6702046),
        (5, 10, 0.95, 0.5, 0.183605591, 0.7531741),
        (6, 10, 0.95, 0.6, 0.252668897, 0.8272210),
        (7, 10, 0.95, 0.7, 0.328716593, 0.8919490),
        (8, 10, 0.95, 0.8, 0.408690782, 0.9458726),
        (9, 10, 0.95, 0.9, 0.473009271, 0.9852814),
        (10, 10, 0.95, 1.0, 0.691502892, 1.0000000),
        # 80% confidence level cases
        (0, 10, 0.8, 0.0, 0.00000000, 0.2056718),
        (1, 10, 0.8, 0.1, 0.02015794, 0.2571711),
        (2, 10, 0.8, 0.2, 0.06973149, 0.3780765),
        (3, 10, 0.8, 0.3, 0.13354110, 0.4867664),
        (4, 10, 0.8, 0.4, 0.20700424, 0.5868026),
        (5, 10, 0.8, 0.5, 0.28829180, 0.6795776),
        (6, 10, 0.8, 0.6, 0.37669356, 0.7654657),
        (7, 10, 0.8, 0.7, 0.47215963, 0.8440668),
        (8, 10, 0.8, 0.8, 0.57505219, 0.9139369),
        (9, 10, 0.8, 0.9, 0.68402740, 0.9711917),
        (10, 10, 0.8, 1.0, 0.79432823, 1.0000000),
    ]


@mark.cloglog
class TestCloglogInterval:
    """Test suite for the cloglog confidence interval."""
    
    def test_cloglog_calculations(self, cloglog_output):
        """Test cloglog interval calculations against known reference values."""
        for x, n, conf_level, expected_mean, expected_lower, expected_upper in cloglog_output:
            result = cloglog(x, n, confidence_level=conf_level)
            
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
        result = cloglog(5, 10)
        assert result.point_estimate == 0.5
        assert result.lower < result.point_estimate < result.upper
        assert result.confidence_level == 0.95

    def test_zero_successes(self):
        """Test edge case with zero successes."""
        result = cloglog(0, 10)
        assert result.point_estimate == 0.0
        assert result.lower == 0.0
        assert result.upper > 0.0  # cloglog should provide non-zero upper bound for x=0

    def test_zero_failures(self):
        """Test edge case with zero failures (all successes)."""
        result = cloglog(10, 10)
        assert result.point_estimate == 1.0
        assert result.lower < 1.0  
        assert result.upper == 1.0
