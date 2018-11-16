# -*- coding: utf-8 -*-
"""
Numerics Functions
------------------
"""
from functools import partial

import numpy as np

from .array_stream import array_stream


@array_stream
def _ireduce_linalg(arrays, func, **kwargs):
    """
    Yield the cumulative reduction of a linag algebra function
    """
    arrays = iter(arrays)
    first = next(arrays)
    second = next(arrays)

    func = partial(func, **kwargs)

    accumulator = func(first, second)
    yield accumulator

    for array in arrays:
        func(accumulator, array, out=accumulator)
        yield accumulator


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
    yield from _ireduce_linalg(arrays, np.dot)


def itensordot(arrays, axes=2):
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
    yield from _ireduce_linalg(arrays, np.tensordot, axes=axes)


def iinner(arrays):
    """
    Cumulative inner product of all arrays in a stream.
    
    Parameters
    ----------
    arrays : iterable
        Arrays to be reduced.
    
    Yields
    ------
    online_inner : ndarray or scalar
    """
    yield from _ireduce_linalg(arrays, np.inner)


def ieinsum(arrays, subscripts, **kwargs):
    """
    Evaluates the Einstein summation convention on the operands.

    Using the Einstein summation convention, many common multi-dimensional 
    array operations can be represented in a simple fashion.

    Parameters
    ----------
    arrays : iterable
        Arrays to be reduced.
    subscripts : str
        Specifies the subscripts for summation.
    dtype : numpy.dtype or None, optional
        The type of the yielded array and of the accumulator in which the elements 
        are combined. The dtype of a is used by default unless a has an integer dtype 
        of less precision than the default platform integer. In that case, if a is 
        signed then the platform integer is used while if a is unsigned then an 
        unsigned integer of the same precision as the platform integer is used.
    order : {'C', 'F', 'A', 'K'}, optional
        Controls the memory layout of the output. 'C' means it should
        be C contiguous. 'F' means it should be Fortran contiguous,
        'A' means it should be 'F' if the inputs are all 'F', 'C' otherwise.
        'K' means it should be as close to the layout as the inputs as
        is possible, including arbitrarily permuted axes.
        Default is 'K'.
    casting : {'no', 'equiv', 'safe', 'same_kind', 'unsafe'}, optional
        Controls what kind of data casting may occur.  Setting this to
        'unsafe' is not recommended, as it can adversely affect accumulations.

          * 'no' means the data types should not be cast at all.
          * 'equiv' means only byte-order changes are allowed.
          * 'safe' means only casts which can preserve values are allowed.
          * 'same_kind' means only safe casts or casts within a kind,
            like float64 to float32, are allowed.
          * 'unsafe' means any data conversions may be done.

        Default is 'safe'.
    optimize : {False, True, 'greedy', 'optimal'}, optional
        Controls if intermediate optimization should occur. No optimization
        will occur if False and True will default to the 'greedy' algorithm.
        Also accepts an explicit contraction list from the ``np.einsum_path``
        function. See ``np.einsum_path`` for more details. Default is False.
    
    Yields
    ------
    online_einsum : ndarray
        Cumulative Einstein summation
    """
    yield from _ireduce_linalg(arrays, partial(np.einsum, subscripts), **kwargs)
