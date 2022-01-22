import pytest
import pandas as pd
import numpy as np
from pynomial.intervals import agresti_coull, asymptotic, bayes, loglog, wilson, exact

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
    

    def test_wilson(self):

        results = binom_methods.get_group('wilson')
        x = results.x.values
        n = results.n.values
        conf = results.conf.values

        target_output = results.loc[:, ['mean', 'lower','upper']].values
        pynomial_output = wilson(x=x, n=n, conf=conf).values

        np.testing.assert_allclose(pynomial_output, target_output)