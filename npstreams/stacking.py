# -*- coding: utf-8 -*-
"""
Stacking arrays from a stream
-----------------------------
"""
from collections.abc import Sized
from functools import partial

import numpy as np

from .array_stream import array_stream


@array_stream
def stack(arrays, axis=-1):
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
    # Shortcut : if axis == -1, this is exactly what ArrayStream.__array__
    if axis == -1:
        return np.array(arrays)

    # TODO: Shortcut if we already know the stream length
    # Note : we are guaranteed that `arrays` is a stream of arrays
    # at worst a tuple (arr,)
    # Use npstreams.length_hint
    arrays = iter(arrays)
    first = next(arrays)
    stack = np.array(first, copy=True)

    for array in arrays:
        stack = np.concatenate([stack, array], axis=axis)

    return stack
