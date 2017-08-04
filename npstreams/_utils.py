# -*- coding: utf-8 -*-
"""
Utilities
---------
"""
import numpy as np
from functools import wraps

def array_stream(func):
    """ Decorates streaming functions to make sure that the stream
    is a stream of ndarrays. If the stream is in fact a single ndarray, 
    this ndarray is repackaged into a sequence 
    
    Parameters
    ----------
    func : callable
        The first argument of `func` must be a stream of arrays.
    """
    @wraps(func)    # thanks functools
    def decorated(arrays, *args, **kwargs):
        if isinstance(arrays, np.ndarray):
            arrays = (arrays,)
        return func(map(np.asarray, arrays), *args, **kwargs)
    return decorated