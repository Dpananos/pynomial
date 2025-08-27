from pytest import mark, fixture
import numpy as np
from pynomial.intervals import logit

TOLERANCE = 1e-6

@fixture
def logit_output():
    """
    Test data for the logit confidence interval.
    Generated from R's binom library using:
    
    binom::binom.logit(0:10, 10, conf.level = 0.95)
    binom::binom.logit(0:10, 10, conf.level = 0.8)
    """
    return [
        # 95% confidence level cases
        (0, 10, 0.95, 0.0, 0.00000000, 0.3084971),
        (1, 10, 0.95, 0.1, 0.01388166, 0.4672367),
        (2, 10, 0.95, 0.2, 0.05041281, 0.5407080),
        (3, 10, 0.95, 0.3, 0.09976832, 0.6236819),
        (4, 10, 0.95, 0.4, 0.15834201, 0.7025951),
        (5, 10, 0.95, 0.5, 0.22450735, 0.7754927),
        (6, 10, 0.95, 0.6, 0.29740491, 0.8416580),
        (7, 10, 0.95, 0.7, 0.37631807, 0.9002317),
        (8, 10, 0.95, 0.8, 0.45929200, 0.9495872),
        (9, 10, 0.95, 0.9, 0.53276327, 0.9861183),
        (10, 10, 0.95, 1.0, 0.69150289, 1.0000000),
        # 80% confidence level cases
        (0, 10, 0.8, 0.0, 0.00000000, 0.2056718),
        (1, 10, 0.8, 0.1, 0.02797423, 0.3001990),
        (2, 10, 0.8, 0.2, 0.08321466, 0.4077828),
        (3, 10, 0.8, 0.3, 0.15037633, 0.5092631),
        (4, 10, 0.8, 0.4, 0.22570867, 0.6039074),
        (5, 10, 0.8, 0.5, 0.30777877, 0.6922212),
        (6, 10, 0.8, 0.6, 0.39609260, 0.7742913),
        (7, 10, 0.8, 0.7, 0.49073688, 0.8496237),
        (8, 10, 0.8, 0.8, 0.59221723, 0.9167853),
        (9, 10, 0.8, 0.9, 0.69980104, 0.9720258),
        (10, 10, 0.8, 1.0, 0.79432823, 1.0000000),
    ]


@mark.logit
class TestLogitInterval:
    """Test suite for the logit confidence interval."""
    
    def test_logit_calculations(self, logit_output):
        """Test logit interval calculations against known R reference values."""
        for x, n, conf_level, expected_mean, expected_lower, expected_upper in logit_output:
            result = logit(x, n, confidence_level=conf_level)
            
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
        result = logit(5, 10)
        assert result.point_estimate == 0.5
        assert result.lower < result.point_estimate < result.upper
        assert result.confidence_level == 0.95

    def test_confidence_levels(self):
        """Test different confidence levels."""
        result_90 = logit(5, 10, confidence_level=0.90)
        result_95 = logit(5, 10, confidence_level=0.95)
        result_99 = logit(5, 10, confidence_level=0.99)
        
        # Same point estimate
        assert np.isclose(result_90.point_estimate, result_95.point_estimate, rtol=TOLERANCE)
        assert np.isclose(result_95.point_estimate, result_99.point_estimate, rtol=TOLERANCE)
        
        # Intervals should get wider with higher confidence
        width_90 = result_90.upper - result_90.lower
        width_95 = result_95.upper - result_95.lower
        width_99 = result_99.upper - result_99.lower
        
        assert width_90 < width_95 < width_99

    def test_logit_transformation_properties(self):
        """Test properties specific to logit transformation."""
        # Test multiple cases
        test_cases = [(2, 10), (3, 10), (5, 10), (7, 10), (8, 10)]
        
        for x, n in test_cases:
            result = logit(x, n, confidence_level=0.95)
            
            # Verify interval is valid
            assert 0 <= result.lower <= result.point_estimate <= result.upper <= 1
            assert result.lower < result.upper  # Should be a proper interval
            
            # Logit transformation should generally give reasonable intervals
            width = result.upper - result.lower
            assert 0.1 < width < 0.9  # Reasonable width bounds

    def test_edge_case_handling(self):
        """Test edge cases that may require special handling."""
        # The logit function has issues with x=0 and x=n due to log(0) and log(1-1)
        # These should either be handled gracefully or raise appropriate errors
        
        # Test x=0 case
        try:
            result_zero = logit(0, 10)
            # If it doesn't raise an error, check if it matches R reference
            # R gives (0.0, 0.0, 0.3084971) for x=0, n=10, conf=0.95
            assert result_zero.point_estimate == 0.0
            assert result_zero.lower == 0.0
            assert np.isclose(result_zero.upper, 0.3084971, rtol=1e-5)
        except (ValueError, ZeroDivisionError):
            # It's acceptable for logit to fail on edge cases
            pass
        
        # Test x=n case  
        try:
            result_all = logit(10, 10)
            # R gives (1.0, 0.69150289, 1.0) for x=10, n=10, conf=0.95
            assert result_all.point_estimate == 1.0
            assert np.isclose(result_all.lower, 0.69150289, rtol=1e-5)
            assert result_all.upper == 1.0
        except (ValueError, ZeroDivisionError):
            # It's acceptable for logit to fail on edge cases
            pass
