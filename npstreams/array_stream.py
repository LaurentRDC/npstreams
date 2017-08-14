# -*- coding: utf-8 -*-
"""
Utilities
---------
"""
import numpy as np
from functools import wraps, partial
from .parallel import pmap

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

# pmap does not support local functions
def _pipe(funcs, array):
    for func in funcs:
        array = func(array)
    return array

def ipipe(*args, **kwargs):
    """
    Pipe arrays through a sequence of functions. For example:

    ``pipe(f, g, h, stream)`` is equivalent to ::

        for arr in stream:
            yield f(g(h(arr)))
    
    Parameters
    ----------
    *funcs : callable
        Callable that support Numpy arrays in their first argument. These
        should *NOT* be generator functions.
    arrays : iterable
        Stream of arrays to be passed.
    processes : int or None, optional, keyword-only
        Number of processes to use. If `None`, maximal number of processes
        is used. Default is one.
    ntotal : int or None, optional, keyword-only
        If the length of `arrays` is known, but passing `arrays` as a list
        would take too much memory, the total number of arrays `ntotal` can be specified. This
        allows for `pmap` to chunk better in case of ``processes > 1``.
    
    Yields
    ------
    piped : ndarray
    """
    arrays = map(np.asarray, args[-1])
    functions = tuple(reversed(args[:-1]))
    yield from pmap(partial(_pipe, functions), arrays, **kwargs)