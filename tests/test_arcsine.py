from pytest import mark, fixture
import numpy as np
from pynomial.intervals import arcsine

TOLERANCE = 1e-6

@fixture
def arcsine_output():
    """
    Test data for the arcsine confidence interval.
    Generated from R's DescTools library using:
    
    DescTools::BinomCI(0:10, 10, method = 'arcsine', conf.level = 0.95)
    DescTools::BinomCI(0:10, 10, method = 'arcsine', conf.level = 0.8)
    """
    return [
        # 95% confidence level cases
        (0, 10, 0.95, 0.0, 0.014815710, 0.2279773),
        (1, 10, 0.95, 0.1, 0.003115191, 0.3911192),
        (2, 10, 0.95, 0.2, 0.031850990, 0.5138248),
        (3, 10, 0.95, 0.3, 0.078978930, 0.6181383),
        (4, 10, 0.95, 0.4, 0.138915842, 0.7096428),
        (5, 10, 0.95, 0.5, 0.209565835, 0.7904342),
        (6, 10, 0.95, 0.6, 0.290357244, 0.8610842),
        (7, 10, 0.95, 0.7, 0.381861733, 0.9210211),
        (8, 10, 0.95, 0.8, 0.486175195, 0.9681490),
        (9, 10, 0.95, 0.9, 0.608880797, 0.9968848),
        (10, 10, 0.95, 1.0, 0.772022717, 0.9851843),
        # 80% confidence level cases
        (0, 10, 0.8, 0.0, 0.0002177226, 0.1448994),
        (1, 10, 0.8, 0.1, 0.0263695545, 0.2897242),
        (2, 10, 0.8, 0.2, 0.0799672136, 0.4071031),
        (3, 10, 0.8, 0.3, 0.1460483908, 0.5119985),
        (4, 10, 0.8, 0.4, 0.2208235554, 0.6081999),
        (5, 10, 0.8, 0.5, 0.3028701375, 0.6971299),
        (6, 10, 0.8, 0.6, 0.3918001244, 0.7791764),
        (7, 10, 0.8, 0.7, 0.4880015288, 0.8539516),
        (8, 10, 0.8, 0.8, 0.5928969206, 0.9200328),
        (9, 10, 0.8, 0.9, 0.7102758305, 0.9736304),
        (10, 10, 0.8, 1.0, 0.8551005676, 0.9997823),
    ]


@mark.arcsine
class TestArcsineInterval:
    """Test suite for the arcsine confidence interval."""
    
    def test_arcsine_calculations(self, arcsine_output):
        """Test arcsine interval calculations against known R reference values."""
        for x, n, conf_level, expected_mean, expected_lower, expected_upper in arcsine_output:
            result = arcsine(x, n, confidence_level=conf_level)
            
            # Test point estimate (mean)
            assert np.isclose(result.point_estimate, expected_mean, rtol=TOLERANCE), \
                f"Point estimate mismatch for x={x}, n={n}, conf_level={conf_level}: got {result.point_estimate}, expected {expected_mean}"
            
            # Test lower bound - be lenient with very small values near zero
            if abs(expected_lower) < 1e-6:
                assert abs(result.lower - expected_lower) < 1e-6, \
                    f"Lower bound mismatch for x={x}, n={n}, conf_level={conf_level}: got {result.lower}, expected {expected_lower}"
            else:
                assert np.isclose(result.lower, expected_lower, rtol=TOLERANCE, atol=1e-6), \
                    f"Lower bound mismatch for x={x}, n={n}, conf_level={conf_level}: got {result.lower}, expected {expected_lower}"
            
            # Test upper bound
            assert np.isclose(result.upper, expected_upper, rtol=TOLERANCE, atol=1e-6), \
                f"Upper bound mismatch for x={x}, n={n}, conf_level={conf_level}: got {result.upper}, expected {expected_upper}"

    def test_basic_calculation(self):
        """Basic calculation test."""
        result = arcsine(5, 10)
        assert result.point_estimate == 0.5
        assert result.lower < result.point_estimate < result.upper
        assert result.confidence_level == 0.95

    def test_zero_successes(self):
        """Test edge case with zero successes."""
        result = arcsine(0, 10)
        assert result.point_estimate == 0.0
        assert result.lower >= 0.0  # Should be non-negative
        assert result.upper > 0.0

    def test_zero_failures(self):
        """Test edge case with zero failures (all successes)."""
        result = arcsine(10, 10)
        assert result.point_estimate == 1.0
        assert result.lower < 1.0
        assert result.upper <= 1.0  # Should not exceed 1

    