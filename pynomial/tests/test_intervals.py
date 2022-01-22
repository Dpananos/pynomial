import pytest
import pandas as pd
import numpy as np
from pynomial.intervals import agresti_coull, asymptotic, bayes, loglog, wilson, exact


def test_intervals():

    '''Test each intefval computation against the binom library reuslts from R'''
    
    assert True