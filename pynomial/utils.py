import numpy as np


def _check_args(x, n, conf):

    x_not_int = (isinstance(x, float) and not x.is_integer()) 
    x_not_int_arr = (isinstance(x, np.ndarray) and np.any(np.mod(x, 1)!=0)) 

    n_not_int = (isinstance(n, float) and not n.is_integer()) 
    n_not_int_arr = (isinstance(n, np.ndarray) and np.any(np.mod(n, 1)!=0))



    if x_not_int or x_not_int_arr:
        raise TypeError("x must be an integer or an array of integers")

    if n_not_int or n_not_int_arr:
        raise TypeError("n must be an integer or an array of integers")

    elif np.any(x < 0) or np.any(n < x) or np.any(n==0):
        raise ValueError(f"x must be a non-negative integer less than or equal to n")

    elif np.any(conf < 0) or np.any(conf > 1):
        raise ValueError("conf must be a number between 0 and 1")

