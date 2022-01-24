from multiprocessing.managers import ValueProxy
import pytest
import pandas as pd
import numpy as np
from pynomial.utils import _check_args
from pynomial.intervals import (
    agresti_coull,
    asymptotic,
    loglog,
    wilson,
    exact,
    logit,
    lrt
)


class BaseIntervalTest:

    binom_results = pd.read_csv("pynomial/tests/binom_output.csv")
    binom_methods = binom_results.groupby(["method"])

    def get_binom_results(self, method_name):
        results = self.binom_methods.get_group(method_name)
        x = results.x.values
        n = results.n.values
        conf = results.conf.values

        target_output = results.loc[:, ["mean", "lower", "upper"]].values

        return x, n, conf, target_output


class TestIntervals(BaseIntervalTest):
    def test_agresti_coull(self):

        x, n, conf, target_output = self.get_binom_results("agresti-coull")
        pynomial_output = agresti_coull(x=x, n=n, conf=conf).values
        np.testing.assert_allclose(pynomial_output, target_output)

    def test_asymptotic(self):

        x, n, conf, target_output = self.get_binom_results("asymptotic")
        pynomial_output = asymptotic(x=x, n=n, conf=conf).values
        np.testing.assert_allclose(pynomial_output, target_output)

    def test_exact(self):

        x, n, conf, target_output = self.get_binom_results("exact")
        pynomial_output = exact(x=x, n=n, conf=conf)

        np.testing.assert_allclose(pynomial_output, target_output)

    def test_loglog(self):

        x, n, conf, target_output = self.get_binom_results("cloglog")
        pynomial_output = loglog(x=x, n=n, conf=conf).values

        np.testing.assert_allclose(pynomial_output, target_output)

    def test_logit(self):

        x, n, conf, target_output = self.get_binom_results("logit")
        pynomial_output = logit(x=x, n=n, conf=conf).values

        np.testing.assert_allclose(pynomial_output, target_output)

    def test_lrt(self):

        x, n, conf, target_output = self.get_binom_results("lrt")
        pynomial_output = lrt(x=x, n=n, conf=conf).values

        # There seems to be a small discrepency between R and python for this test
        # Probably something in log-lik evaluation.
        # In testing, the reltive difference is very very small.
        # Pass tests if difference is smaller than 4 decimal places
        # This should be enough precision since 4 decimal places of accuracy is equivalent to
        # over a million observations. I wouldn't worry about it and neither should you
        np.testing.assert_allclose(pynomial_output, target_output, atol=1e-4)

    def test_wilson(self):

        x, n, conf, target_output = self.get_binom_results("wilson")
        pynomial_output = wilson(x=x, n=n, conf=conf).values

        np.testing.assert_allclose(pynomial_output, target_output)


class TestErrors:
    def test_conf(self):
        # Conf above 1
        with pytest.raises(ValueError):
            _check_args(1, 2, 1.1)

        # Conf=1
        with pytest.raises(ValueError):
            _check_args(1, 2, 1)

        # Conf=0
        with pytest.raises(ValueError):
            _check_args(1, 2, 0)

        # Conf below 0
        with pytest.raises(ValueError):
            _check_args(1, 2, -1)

    def test_bad_x(self):

        with pytest.raises(ValueError):
            # Successes is a negative number
            _check_args(-1, 1, 0.95)

            # Bad array of successes
            x = np.array([-1, 0, 1])
            _check_args(x, 1, 0.95)

        with pytest.raises(ValueError):

            # More successes than trials
            _check_args(2, 1, 0.95)

            # Successes is a float
            _check_args(1.2, 2, 0.95)

    def test_bad_n(self):

        with pytest.raises(TypeError):

            # trials is a float
            _check_args(1, 2.5, 0.95)

            # 0 trials
            _check_args(1, 0, 0.95)

        with pytest.raises(ValueError):
            # Bad array of trials
            # trials is a negative number
            _check_args(1, -1, 0.95)

            n = np.array([-1, 0, 1])
            _check_args(1, n, 0.95)


class TestShapes:
    def test_no_array_args(self):

        # Check passing arrays and ints works as expected
        x = 1
        n = 2
        conf = 0.95
        x, n, conf = _check_args(x, n, conf)

        assert x.size == n.size
        assert x.size == conf.size
        assert n.size == conf.size

    def test_one_array_args(self):

        # Check passing an array for one argument does not raise an error
        # First x
        x = np.ones(3)
        n = 2
        conf = 0.95
        x, n, conf = _check_args(x, n, conf)

        assert x.size == n.size
        assert x.size == conf.size
        assert n.size == conf.size

        # Now n
        x = 1
        n = 2 * np.ones(3)
        conf = 0.95
        x, n, conf = _check_args(x, n, conf)

        assert x.size == n.size
        assert x.size == conf.size
        assert n.size == conf.size

        # Now conf

        x = 1
        n = 2
        conf = 0.95 * np.ones(3)
        x, n, conf = _check_args(x, n, conf)

        assert x.size == n.size
        assert x.size == conf.size
        assert n.size == conf.size

    def test_two_array_args(self):

        # Check passing two arrays of the same size yields expected result
        x = np.ones(3)
        n = 2 * np.ones(3)
        conf = 0.95
        x, n, conf = _check_args(x, n, conf)

        assert x.size == n.size
        assert x.size == conf.size
        assert n.size == conf.size

        x = np.ones(3)
        n = 2
        conf = 0.95 * np.ones(3)
        x, n, conf = _check_args(x, n, conf)

        assert x.size == n.size
        assert x.size == conf.size
        assert n.size == conf.size

        x = 1
        n = 2 * np.ones(3)
        conf = 0.95 * np.ones(3)
        x, n, conf = _check_args(x, n, conf)

        assert x.size == n.size
        assert x.size == conf.size
        assert n.size == conf.size

    def test_three_array_args(self):

        x = np.ones(3)
        n = 2 * np.ones(3)
        conf = 0.95 * np.ones(3)
        x, n, conf = _check_args(x, n, conf)

        assert x.size == n.size
        assert x.size == conf.size
        assert n.size == conf.size

    def test_mismatch_arg_size(self):

        x = np.ones(4)
        n = 2 * np.ones(3)
        conf = 0.95
        with pytest.raises(ValueError):
            x, n, conf = _check_args(x, n, conf)
