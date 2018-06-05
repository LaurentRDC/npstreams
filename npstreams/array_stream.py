# -*- coding: utf-8 -*-
"""
Utilities
---------
"""
from collections.abc import Iterator
from functools import partial, wraps
from numpy import asanyarray

import numpy as np

class ArrayStream(Iterator):
    """ 
    Iterator of arrays. Elements from the stream are converted to 
    NumPy arrays. If ``stream`` is a single array, it will be 
    repackaged as a length 1 iterable.

    .. versionadded:: 1.6
    """

    def __init__(self, stream):
        if isinstance(stream, np.ndarray):
            stream = (stream,)
        self._iterator = iter(stream)
    
    def __next__(self):
        n = next(self._iterator)
        return asanyarray(n)

def array_stream(func):
    """ 
    Decorates streaming functions to make sure that the stream
    is a stream of ndarrays. Objects that are not arrays are transformed 
    into arrays. If the stream is in fact a single ndarray, this ndarray 
    is repackaged into a sequence of length 1.

    The first argument of the decorated function is assumed to be an iterable of
    arrays, or an iterable of objects that can be casted to arrays.
    """
    @wraps(func)
    def decorated(arrays, *args, **kwargs):
        return func(ArrayStream(arrays), *args, **kwargs)
    return decorated
