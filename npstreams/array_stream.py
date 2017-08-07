# -*- coding: utf-8 -*-
"""
Utilities
---------
"""
import numpy as np
from functools import wraps

def array_stream(func):
    """ 
    Decorates streaming functions to make sure that the stream
    is a stream of ndarrays. Objects that are not arrays
    are transformed into arrays using ``numpy.asarray``. If the stream 
    is in fact a single ndarray, this ndarray is repackaged into a sequence of
    length 1.
    
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

# TODO: keyword-only argument 'processes' as a proxy for pmap?
@array_stream
def ipipe(arrays, *funcs):
    """
    Pipe arrays through a sequence of functions. For example:

    ``pipe(stream, f, g, h)`` is roughly equivalent to ::

        for arr in stream:
            yield f(g(h(arr)))
    
    Parameters
    ----------
    arrays : iterable
        Arrays
    funcs : callable
        Callable that support Numpy arrays in their first argument.
    
    Yield
    -----
    piped : ndarray
    """
    functions = tuple(reversed(funcs))
    def pipe(arr):
        for func in functions:
            arr = func(arr)
        return arr
    yield from map(pipe, arrays)