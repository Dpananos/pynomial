from pytest import mark, fixture
import numpy as np
from pynomial.intervals import agresti_coull

TOLERANCE = 1e-6

@fixture
def agresti_coull_output():
    """
    Test data for the Agresti-Coull confidence interval.
    Generated from R's binom library using the following code:

    #> binom::binom.agresti.coull(0:10, 10, conf.level = 0.95)
    #> binom::binom.agresti.coull(0:10, 10, conf.level = 0.8)
    """
    return [
        # 95% confidence level cases
        (0, 10, 0.95, 0.0, -0.043354506, 0.3208873),
        (1, 10, 0.95, 0.1, -0.003941498, 0.4259677),
        (2, 10, 0.95, 0.2, 0.045887270, 0.5206324),
        (3, 10, 0.95, 0.3, 0.103338418, 0.6076747),
        (4, 10, 0.95, 0.4, 0.167110626, 0.6883959),
        (5, 10, 0.95, 0.5, 0.236593091, 0.7634069),
        (6, 10, 0.95, 0.6, 0.311604066, 0.8328894),
        (7, 10, 0.95, 0.7, 0.392325298, 0.8966616),
        (8, 10, 0.95, 0.8, 0.479367591, 0.9541127),
        (9, 10, 0.95, 0.9, 0.574032263, 1.0039415),
        (10, 10, 0.95, 1.0, 0.679112694, 1.0433545),
        # 80% confidence level cases
        (0, 10, 0.8, 0.0, -0.02563404, 0.1667027),
        (1, 10, 0.8, 0.1, 0.01999020, 0.2928647),
        (2, 10, 0.8, 0.2, 0.08138460, 0.4032566),
        (3, 10, 0.8, 0.3, 0.15185002, 0.5045775),
        (4, 10, 0.8, 0.4, 0.22910304, 0.5991107),
        (5, 10, 0.8, 0.5, 0.31220444, 0.6877956),
        (6, 10, 0.8, 0.6, 0.40088930, 0.7708970),
        (7, 10, 0.8, 0.7, 0.49542255, 0.8481500),
        (8, 10, 0.8, 0.8, 0.59674339, 0.9186154),
        (9, 10, 0.8, 0.9, 0.70713525, 0.9800098),
        (10, 10, 0.8, 1.0, 0.83329728, 1.0256340),
    ]


@mark.agresti_coull
class TestAgrestiCoullInterval:
    """Test suite for the Agresti-Coull confidence interval."""
    
    def test_agresti_coull_calculations(self, agresti_coull_output):
        """Test Agresti-Coull interval calculations against known reference values."""
        for x, n, conf_level, expected_mean, expected_lower, expected_upper in agresti_coull_output:
            result = agresti_coull(x, n, confidence_level=conf_level)

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
        result = agresti_coull(5, 10)
        # For Agresti-Coull, point estimate is adjusted, not raw proportion
        assert result.lower < result.point_estimate < result.upper
        assert result.confidence_level == 0.95

    
    def test_zero_successes(self):
        """Test edge case with zero successes."""
        result = agresti_coull(0, 10)
        assert result.point_estimate == 0.0
        assert result.lower <= 0.0
        assert result.upper > 0.0

    def test_zero_failures(self):
        """Test edge case with zero failures (all successes)."""
        result = agresti_coull(10, 10)
        assert result.point_estimate == 1.0  # Adjusted estimate is less than 1
        assert result.lower < 1.0
        assert result.upper >=1.0
