import pytest
import pandas as pd
import numpy as np
from pynomial.utils import _check_args
from pynomial.intervals import agresti_coull, asymptotic, bayes, loglog, wilson, exact, logit

binom_results = pd.read_csv("pynomial/tests/binom_output.csv")
binom_methods = binom_results.groupby(['method'])

class TestIntervals:

    def test_agresti_coull(self):

        results = binom_methods.get_group('agresti-coull')
        x = results.x.values
        n = results.n.values
        conf = results.conf.values

        target_output = results.loc[:, ['mean', 'lower','upper']].values
        pynomial_output = agresti_coull(x=x, n=n, conf=conf).values

        np.testing.assert_allclose(pynomial_output, target_output)
    
    def test_asymptotic(self):

        results = binom_methods.get_group('asymptotic')
        x = results.x.values
        n = results.n.values
        conf = results.conf.values

        target_output = results.loc[:, ['mean', 'lower','upper']].values
        pynomial_output = asymptotic(x=x, n=n, conf=conf).values

        np.testing.assert_allclose(pynomial_output, target_output)


    def test_exact(self):

        results = binom_methods.get_group('exact')
        x = results.x.values
        n = results.n.values
        conf = results.conf.values

        target_output = results.loc[:, ['mean', 'lower','upper']]
        pynomial_output = exact(x=x, n=n, conf=conf)

        np.testing.assert_allclose(pynomial_output, target_output)


    def test_loglog(self):

        results = binom_methods.get_group('cloglog')
        x = results.x.values
        n = results.n.values
        conf = results.conf.values

        target_output = results.loc[:, ['mean', 'lower','upper']].values
        pynomial_output = loglog(x=x, n=n, conf=conf).values

        np.testing.assert_allclose(pynomial_output, target_output)


    def test_logit(self):

        results = binom_methods.get_group('logit')
        x = results.x.values
        n = results.n.values
        conf = results.conf.values

        target_output = results.loc[:, ['mean', 'lower','upper']].values
        pynomial_output = logit(x=x, n=n, conf=conf).values

        np.testing.assert_allclose(pynomial_output, target_output)
    
    

    def test_wilson(self):

        results = binom_methods.get_group('wilson')
        x = results.x.values
        n = results.n.values
        conf = results.conf.values

        target_output = results.loc[:, ['mean', 'lower','upper']].values
        pynomial_output = wilson(x=x, n=n, conf=conf).values

        np.testing.assert_allclose(pynomial_output, target_output)

class TestErrors:

    def test_conf(self):
        # Conf above 1
        with pytest.raises(ValueError):
            _check_args(1, 2, 1.1)

        # Conf below 0
        with pytest.raises(ValueError):
            _check_args(1, 2, -1)

    def test_bad_x(self):

        with pytest.raises(ValueError):
        # Successes is a negative number
            _check_args(-1, 1, 0.95)

        # More successes than trials
            _check_args(2, 1, 0.95)
        
        # Successes is a float
            _check_args(1.2, 2, 0.95)

        # Bad array of successes
            x = np.array([-1, 0, 1])
            _check_args(x, 1, 0.95) 


    def test_bad_n(self):

        with pytest.raises(ValueError):
        # trials is a negative number
            _check_args(1, -1, 0.95)
        
        # trials is a float
            _check_args(1, 2.5, 0.95)

        # Bad array of trials
            n = np.array([-1, 0, 1])
            _check_args(1, n, 0.95) 
