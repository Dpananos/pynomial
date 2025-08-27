from pytest import mark, fixture
import numpy as np
from pynomial.intervals import clopper_pearson

TOLERANCE = 1e-6

@fixture
def clopper_pearson_output():
    """
    Test data for the Clopper-Pearson (exact) confidence interval.
    Generated from R's binom library using:
    
    binom::binom.exact(0:10, 10, conf.level = 0.95)
    binom::binom.exact(0:10, 10, conf.level = 0.8)
    """
    return [
        # 95% confidence level cases
        (0, 10, 0.95, 0.0, 0.000000000, 0.3084971),
        (1, 10, 0.95, 0.1, 0.002528579, 0.4450161),
        (2, 10, 0.95, 0.2, 0.025210726, 0.5560955),
        (3, 10, 0.95, 0.3, 0.066739511, 0.6524529),
        (4, 10, 0.95, 0.4, 0.121552258, 0.7376219),
        (5, 10, 0.95, 0.5, 0.187086028, 0.8129140),
        (6, 10, 0.95, 0.6, 0.262378077, 0.8784477),
        (7, 10, 0.95, 0.7, 0.347547150, 0.9332605),
        (8, 10, 0.95, 0.8, 0.443904538, 0.9747893),
        (9, 10, 0.95, 0.9, 0.554983883, 0.9974714),
        (10, 10, 0.95, 1.0, 0.691502892, 1.0000000),
        # 80% confidence level cases
        (0, 10, 0.8, 0.0, 0.00000000, 0.2056718),
        (1, 10, 0.8, 0.1, 0.01048074, 0.3368477),
        (2, 10, 0.8, 0.2, 0.05452862, 0.4496039),
        (3, 10, 0.8, 0.3, 0.11582528, 0.5517308),
        (4, 10, 0.8, 0.4, 0.18756230, 0.6457841),
        (5, 10, 0.8, 0.5, 0.26731810, 0.7326819),
        (6, 10, 0.8, 0.6, 0.35421593, 0.8124377),
        (7, 10, 0.8, 0.7, 0.44826917, 0.8841747),
        (8, 10, 0.8, 0.8, 0.55039611, 0.9454714),
        (9, 10, 0.8, 0.9, 0.66315228, 0.9895193),
        (10, 10, 0.8, 1.0, 0.79432823, 1.0000000),
    ]


@mark.clopper_pearson
class TestClopperPearsonInterval:
    """Test suite for the Clopper-Pearson (exact) confidence interval."""
    
    def test_exact_calculations(self, clopper_pearson_output):
        """Test Clopper-Pearson interval calculations against known R reference values."""
        for x, n, conf_level, expected_mean, expected_lower, expected_upper in clopper_pearson_output:
            result = clopper_pearson(x, n, confidence_level=conf_level)
            
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
        result = clopper_pearson(5, 10)
        assert result.point_estimate == 0.5
        assert result.lower < result.point_estimate < result.upper
        assert result.confidence_level == 0.95

    def test_zero_successes(self):
        """Test edge case with zero successes."""
        result = clopper_pearson(0, 10)
        assert result.point_estimate == 0.0
        assert result.lower == 0.0  # Exact method gives exact 0 for lower bound when x=0
        assert result.upper > 0.0

    def test_zero_failures(self):
        """Test edge case with zero failures (all successes)."""
        result = clopper_pearson(10, 10)
        assert result.point_estimate == 1.0
        assert result.lower < 1.0  # Should have some uncertainty even for x=n
        assert result.upper == 1.0  # Exact method gives exact 1 for upper bound when x=n