# -*- coding: utf-8 -*-
"""
Stacking arrays from a stream
-----------------------------
"""
from functools import partial
from collections import Sized

import numpy as np

from .array_stream import array_stream

@array_stream
def stack(arrays, axis = -1):
    """ 
    Stack of all arrays from a stream. Generalization of numpy.stack
    and numpy.concatenate. 

    Parameters
    ----------
    arrays : iterable
        Stream of NumPy arrays. Arrays must have shapes that broadcast together.
    axis : int, optional
        Stacking direction. If ``axis = -1``, arrays are stacked along a
        new dimension.
    
    Returns
    -------
    stacked : ndarray
        Cumulative stacked array.
    """
    arrays = iter(arrays)

    # Shortcut if we already know the stream length
    # Note : we are guaranteed that `arrays` is a stream of arrays
    # at worst a tuple (arr,)
    if isinstance(arrays, Sized):
        return np.stack(arrays, axis = axis)

    first = next(arrays)
    stack = np.array(first, copy = True)

    if axis == -1:
        axis = stack.ndim
        stack = np.expand_dims(stack, axis = axis)
        arrays = map(partial(np.expand_dims, axis = axis), arrays)
    
    for array in arrays:
        stack = np.concatenate([stack, array], axis = axis)

    return stack
