from tkinter import N
import pytest
import pandas as pd
import numpy as np
from pynomial.utils import _check_args


class TestShapes:
    def test_no_array_args(self):

        # Check passing arrays and ints works as expected
        x = 1
        n = 2
        conf = 0.95
        x, n, conf = _check_args(x, n, conf)

        assert len(x) == len(n)
        assert len(x) == len(conf)
        assert len(n) == len(conf)

    def test_one_array_args(self):

        # Check passing an array for one argument does not raise an error
        # First x
        x = np.ones(3)
        n = 2
        conf = 0.95
        x, n, conf = _check_args(x, n, conf)

        assert len(x) == len(n)
        assert len(x) == len(conf)
        assert len(n) == len(conf)

        # Now n
        x = 1
        n = 2 * np.ones(3)
        conf = 0.95
        x, n, conf = _check_args(x, n, conf)

        assert len(x) == len(n)
        assert len(x) == len(conf)
        assert len(n) == len(conf)

        # Now conf

        x = 1
        n = 2
        conf = 0.95 * np.ones(3)
        x, n, conf = _check_args(x, n, conf)

        assert len(x) == len(n)
        assert len(x) == len(conf)
        assert len(n) == len(conf)

    def test_two_array_args(self):

        # Check passing two arrays of the same size yields expected result
        x = np.ones(3)
        n = 2 * np.ones(3)
        conf = 0.95
        x, n, conf = _check_args(x, n, conf)

        assert len(x) == len(n)
        assert len(x) == len(conf)
        assert len(n) == len(conf)

        x = np.ones(3)
        n = 2
        conf = 0.95 * np.ones(3)
        x, n, conf = _check_args(x, n, conf)

        assert len(x) == len(n)
        assert len(x) == len(conf)
        assert len(n) == len(conf)

        x = 1
        n = 2 * np.ones(3)
        conf = 0.95 * np.ones(3)
        x, n, conf = _check_args(x, n, conf)

        assert len(x) == len(n)
        assert len(x) == len(conf)
        assert len(n) == len(conf)

    def test_three_array_args(self):

        x = np.ones(3)
        n = 2 * np.ones(3)
        conf = 0.95 * np.ones(3)
        x, n, conf = _check_args(x, n, conf)
        
        assert len(x) == len(n)
        assert len(x) == len(conf)
        assert len(n) == len(conf)

    def test_mismatch_arg_size(self):

        x = np.ones(4)
        n = 2 * np.ones(3)
        conf = 0.95
        with pytest.raises(ValueError):
            x, n, conf = _check_args(x, n, conf)


class TestOneSidedIntervals:

    @pytest.mark.xfail
    def test_one_sided_intervals(self):
        
        assert False