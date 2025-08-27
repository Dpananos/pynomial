from pytest import mark, fixture
import numpy as np
from pynomial.intervals import wilson

TOLERANCE = 1e-6

@fixture
def wilson_output():
    """
    Test data for the Wilson confidence interval.
    Generated from R's binom library using:
    
    binom::binom.wilson(0:10, 10, conf.level = 0.95)
    binom::binom.wilson(0:10, 10, conf.level = 0.8)
    
    Note: These are without continuity correction (R's default)
    """
    return [
        # 95% confidence level cases
        (0, 10, 0.95, 0.0, 2.005249e-17, 0.2775328),
        (1, 10, 0.95, 0.1, 1.787621e-02, 0.4041500),
        (2, 10, 0.95, 0.2, 5.668215e-02, 0.5098375),
        (3, 10, 0.95, 0.3, 1.077913e-01, 0.6032219),
        (4, 10, 0.95, 0.4, 1.681803e-01, 0.6873262),
        (5, 10, 0.95, 0.5, 2.365931e-01, 0.7634069),
        (6, 10, 0.95, 0.6, 3.126738e-01, 0.8318197),
        (7, 10, 0.95, 0.7, 3.967781e-01, 0.8922087),
        (8, 10, 0.95, 0.8, 4.901625e-01, 0.9433178),
        (9, 10, 0.95, 0.9, 5.958500e-01, 0.9821238),
        (10, 10, 0.95, 1.0, 7.224672e-01, 1.0000000),
        # 80% confidence level cases
        (0, 10, 0.8, 0.0, -1.192007e-17, 0.1410687),
        (1, 10, 0.8, 0.1, 3.041064e-02, 0.2824443),
        (2, 10, 0.8, 0.2, 8.623722e-02, 0.3984040),
        (3, 10, 0.8, 0.3, 1.537992e-01, 0.5026283),
        (4, 10, 0.8, 0.4, 2.295656e-01, 0.5986482),
        (5, 10, 0.8, 0.5, 3.122044e-01, 0.6877956),
        (6, 10, 0.8, 0.6, 4.013518e-01, 0.7704344),
        (7, 10, 0.8, 0.7, 4.973717e-01, 0.8462008),
        (8, 10, 0.8, 0.8, 6.015960e-01, 0.9137628),
        (9, 10, 0.8, 0.9, 7.175557e-01, 0.9695894),
        (10, 10, 0.8, 1.0, 8.589313e-01, 1.0000000),
    ]


@mark.wilson
class TestWilsonInterval:
    """Test suite for the Wilson confidence interval."""
    
    def test_wilson_calculations(self, wilson_output):
        """Test Wilson interval calculations against known R reference values."""
        for x, n, conf_level, expected_mean, expected_lower, expected_upper in wilson_output:
            # Use correct_continuity=False to match R's binom.wilson default
            result = wilson(x, n, confidence_level=conf_level, correct_continuity=False)
            
            # Test point estimate (mean)
            assert np.isclose(result.point_estimate, expected_mean, rtol=TOLERANCE), \
                f"Point estimate mismatch for x={x}, n={n}, conf_level={conf_level}: got {result.point_estimate}, expected {expected_mean}"
            
            # Test lower bound - be lenient with very small values near zero
            if abs(expected_lower) < 1e-10:
                assert abs(result.lower) < 1e-10, \
                    f"Lower bound should be near zero for x={x}, n={n}, conf_level={conf_level}: got {result.lower}, expected {expected_lower}"
            else:
                assert np.isclose(result.lower, expected_lower, rtol=TOLERANCE), \
                    f"Lower bound mismatch for x={x}, n={n}, conf_level={conf_level}: got {result.lower}, expected {expected_lower}"
            
            # Test upper bound
            assert np.isclose(result.upper, expected_upper, rtol=TOLERANCE), \
                f"Upper bound mismatch for x={x}, n={n}, conf_level={conf_level}: got {result.upper}, expected {expected_upper}"

    def test_calculation(self):
        """Basic calculation test."""
        result = wilson(5, 10, correct_continuity=False)
        assert result.point_estimate == 0.5
        assert result.lower < result.point_estimate < result.upper
        assert result.confidence_level == 0.95

    def test_continuity_correction_options(self):
        """Test that both continuity correction options work."""
        # Without continuity correction
        result_no_cc = wilson(5, 10, correct_continuity=False)
        
        # With continuity correction
        result_with_cc = wilson(5, 10, correct_continuity=True)
        
        # Both should have same point estimate
        assert np.isclose(result_no_cc.point_estimate, result_with_cc.point_estimate, rtol=TOLERANCE)
        
        # Intervals should be different (continuity correction generally makes them wider)
        assert result_no_cc.lower != result_with_cc.lower or result_no_cc.upper != result_with_cc.upper

    def test_zero_successes(self):
        """Test edge case with zero successes."""
        result = wilson(0, 10, correct_continuity=False)
        assert result.point_estimate == 0.0
        assert result.lower >= 0.0  # Should be non-negative
        assert result.upper > 0.0

    def test_zero_failures(self):
        """Test edge case with zero failures (all successes)."""
        result = wilson(10, 10, correct_continuity=False)
        assert result.point_estimate == 1.0
        assert result.lower < 1.0
        assert np.isclose(result.upper, 1.0, rtol=TOLERANCE)  # Should be very close to 1

    def test_confidence_levels(self):
        """Test different confidence levels."""
        result_90 = wilson(5, 10, confidence_level=0.90, correct_continuity=False)
        result_95 = wilson(5, 10, confidence_level=0.95, correct_continuity=False)
        result_99 = wilson(5, 10, confidence_level=0.99, correct_continuity=False)
        
        # Same point estimate
        assert np.isclose(result_90.point_estimate, result_95.point_estimate, rtol=TOLERANCE)
        assert np.isclose(result_95.point_estimate, result_99.point_estimate, rtol=TOLERANCE)
        
        # Intervals should get wider with higher confidence
        width_90 = result_90.upper - result_90.lower
        width_95 = result_95.upper - result_95.lower
        width_99 = result_99.upper - result_99.lower
        
        assert width_90 < width_95 < width_99

    def test_wilson_properties(self):
        """Test properties specific to Wilson intervals."""
        # Test multiple cases
        test_cases = [(1, 10), (3, 10), (5, 10), (7, 10), (9, 10)]
        
        for x, n in test_cases:
            result = wilson(x, n, confidence_level=0.95, correct_continuity=False)
            
            # Verify interval is valid
            assert 0 <= result.lower <= result.point_estimate <= result.upper <= 1
            assert result.lower < result.upper  # Should be a proper interval
            
            # Wilson intervals should generally be reasonable
            width = result.upper - result.lower
            assert 0.1 < width < 1.0  # Reasonable width bounds
