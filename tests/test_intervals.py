from pytest import mark, fixture
import numpy as np
from pynomial.intervals import wald, cloglog, agresti_coull, bayesian_beta, clopper_pearson, logit

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
            assert np.isclose(result.lower, expected_lower, rtol=TOLERANCE), \
                f"Lower bound mismatch for x={x}, n={n}, conf_level={conf_level}: got {result.lower}, expected {expected_lower}"
            
            # Test upper bound
            assert np.isclose(result.upper, expected_upper, rtol=TOLERANCE), \
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

    def test_prior_parameters(self):
        """Test different prior parameters."""
        # Uniform prior (1, 1)
        result_uniform = bayesian_beta(5, 10, prior_alpha=1.0, prior_beta=1.0)
        # Posterior mean should be (5 + 1)/(10 + 2) = 6/12 = 0.5
        assert np.isclose(result_uniform.point_estimate, 0.5, rtol=TOLERANCE)
        
        # Strong prior favoring low values
        result_low = bayesian_beta(5, 10, prior_alpha=1.0, prior_beta=10.0)
        # Should pull the estimate lower than the frequentist 0.5
        assert result_low.point_estimate < 0.5

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

    def test_confidence_levels(self):
        """Test different confidence levels."""
        result_90 = bayesian_beta(5, 10, confidence_level=0.90)
        result_95 = bayesian_beta(5, 10, confidence_level=0.95)
        result_99 = bayesian_beta(5, 10, confidence_level=0.99)
        
        # Same point estimate
        assert np.isclose(result_90.point_estimate, result_95.point_estimate, rtol=TOLERANCE)
        assert np.isclose(result_95.point_estimate, result_99.point_estimate, rtol=TOLERANCE)
        
        # Intervals should get wider with higher confidence
        width_90 = result_90.upper - result_90.lower
        width_95 = result_95.upper - result_95.lower
        width_99 = result_99.upper - result_99.lower
        
        assert width_90 < width_95 < width_99


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

    def test_confidence_levels(self):
        """Test different confidence levels."""
        result_90 = clopper_pearson(5, 10, confidence_level=0.90)
        result_95 = clopper_pearson(5, 10, confidence_level=0.95)
        result_99 = clopper_pearson(5, 10, confidence_level=0.99)
        
        # Same point estimate
        assert np.isclose(result_90.point_estimate, result_95.point_estimate, rtol=TOLERANCE)
        assert np.isclose(result_95.point_estimate, result_99.point_estimate, rtol=TOLERANCE)
        
        # Intervals should get wider with higher confidence
        width_90 = result_90.upper - result_90.lower
        width_95 = result_95.upper - result_95.lower
        width_99 = result_99.upper - result_99.lower
        
        assert width_90 < width_95 < width_99

    def test_exact_coverage_properties(self):
        """Test that Clopper-Pearson gives exact coverage (conservative)."""
        # Test multiple cases to ensure intervals are never too narrow
        test_cases = [(1, 10), (3, 10), (7, 10), (9, 10)]
        
        for x, n in test_cases:
            result = clopper_pearson(x, n, confidence_level=0.95)
            
            # Verify interval is valid
            assert 0 <= result.lower <= result.point_estimate <= result.upper <= 1
            assert result.lower < result.upper  # Should be a proper interval
            
            # For exact method, intervals should be reasonably wide for small n
            width = result.upper - result.lower
            assert width > 0.1  # Reasonable width for n=10


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
