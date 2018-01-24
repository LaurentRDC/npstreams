# -*- coding: utf-8 -*-
"""
Stacking arrays from a stream
-----------------------------
"""
from functools import partial

import numpy as np

from . import array_stream

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

    return stack
