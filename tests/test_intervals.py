from pytest import mark, fixture
import numpy as np
from pynomial.intervals import wald, cloglog, agresti_coull

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
