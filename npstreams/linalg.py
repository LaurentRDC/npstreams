# -*- coding: utf-8 -*-
"""
Numerics Functions
------------------
"""

import numpy as np
from . import array_stream

@array_stream
def idot(arrays):
    """
    Yields the cumulative array inner product (dot product) of arrays.

    Parameters
    ----------
    arrays : iterable
        Arrays to be reduced.
    
    Yields
    ------
    online_dot : ndarray

    See Also
    --------
    numpy.linalg.multi_dot : Compute the dot product of two or more arrays in a single function call, 
                             while automatically selecting the fastest evaluation order.
    """
    arrays = iter(arrays)
    first = next(arrays)
    second = next(arrays)

    accumulator = np.dot(first, second)
    yield accumulator

    for array in arrays:
        # For some reason, np.dot(..., out = accumulator) did not produce results
        # that were equal to numpy.linalg.multi_dot
        accumulator[:] = np.dot(accumulator, array)
        yield accumulator

@array_stream
def itensordot(arrays, axes = 2):
    """
    Yields the cumulative array inner product (dot product) of arrays.

    Parameters
    ----------
    arrays : iterable
        Arrays to be reduced.
    axes : int or (2,) array_like
        * integer_like: If an int N, sum over the last N axes of a 
          and the first N axes of b in order. The sizes of the corresponding axes must match.
        * (2,) array_like: Or, a list of axes to be summed over, first sequence applying to a, 
          second to b. Both elements array_like must be of the same length.
    
    Yields
    ------
    online_tensordot : ndarray

    See Also
    --------
    numpy.tensordot : Compute the tensordot on two tensors.
    """
    arrays = iter(arrays)
    first = next(arrays)
    second = next(arrays)

    accumulator = np.tensordot(first, second, axes = axes)
    yield accumulator

    for array in arrays:
        # For some reason, np.dot(..., out = accumulator) did not produce results
        # that were equal to numpy.linalg.multi_dot
        accumulator[:] = np.tensordot(accumulator, array, axes = axes)
        yield accumulator