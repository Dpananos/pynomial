import numpy as np

def _check_args(x,n, conf):

    if (isinstance(x, np.ndarray) and not issubclass(x.dtype.type, np.integer)) or (isinstance(x, float) and not x.is_integer()):
        raise ValueError('x must be an integer or an array of integers')

    elif (isinstance(n, np.ndarray) and not issubclass(n.dtype.type, np.integer)) or (isinstance(n, float) and not n.is_integer()):
        raise ValueError('n must be an integer or an array of integers')

    elif np.any(x<0) or np.any(n<x):
        raise ValueError(f'x must be a non-negative integer less than or equal to n')

    elif np.any(conf<0) or np.any(conf>1):
        raise ValueError('conf must be a number between 0 and 1')

