# -*- coding: utf-8 -*-
"""
Stacking arrays from a stream
-----------------------------
"""
from functools import partial

import numpy as np

from . import array_stream


@array_stream
def istack(arrays, axis = -1):
    """ 
    Stack arrays from a stream. Generalization of numpy.stack
    and numpy.concatenate.

    Parameters
    ----------
    arrays : iterable
        Stream of NumPy arrays.
    axis : int, optional
        Stacking direction. If ``axis = -1``, arrays are stacked along a
        new dimension.
    
    Yields
    ------
    online_stack : ndarray
        Cumulative stacked array.
    """
    arrays = iter(arrays)
    first = next(arrays)

    stack = np.array(first, copy = True)
    if axis == -1:
        axis = stack.ndim
        stack = np.expand_dims(stack, axis = axis)
        arrays = map(partial(np.expand_dims, axis = axis), arrays)
    
    for array in arrays:
        stack = np.concatenate([stack, array], axis = axis)
        yield stack

@array_stream
def iflatten(arrays):
    """
    flatten the arrays in a stream into a single, 1D array. Note that
    the order of flattening is not guaranteed.

    Parameters
    ----------
    arrays : iterable
        Stream of NumPy arrays. Contrary to convention, these
        arrays do not need to be of the same shape. 
    
    Yields
    ------
    online_flatten : ndarray
        Cumulative flattened array.
    """
    arrays = map(np.ravel, arrays)
    yield from istack(arrays, axis = 0)
