# -*- coding: utf-8 -*-
"""
Numerics Functions
------------------
"""

import numpy as np
from functools import partial

# TODO: is in-place justified?
def _nan_to_num(array, fill):
    """ Replace NaNs in `array` with `fill`. Keyword-arguments
    are passed to numpy.nan_to_num"""
    with_nans = np.array(array)
    with_nans[np.isnan(with_nans)] = fill
    return with_nans

def isum(arrays, dtype = None):
    """ 
    Streaming sum of array elements.

    Parameters
    ----------
    arrays : iterable
        Arrays to be summed.
    dtype : numpy.dtype, optional
        The type of the yielded array and of the accumulator in which the elements 
        are summed. The dtype of a is used by default unless a has an integer dtype 
        of less precision than the default platform integer. In that case, if a is 
        signed then the platform integer is used while if a is unsigned then an 
        unsigned integer of the same precision as the platform integer is used.
    
    Yields
    ------
    online_sum : ndarray
    """
    arrays = iter(arrays)

    first = next(arrays)
    if dtype is None:
        dtype = first.dtype
    
    accumulator = first.astype(dtype, copy = True)
    for array in arrays:
        accumulator += array.astype(dtype, copy = False)
        yield accumulator

def inansum(arrays, dtype = None):
    """ 
    Streaming sum of array elements. NaNs are ignored (i.e. treated as zero).

    Parameters
    ----------
    arrays : iterable
        Arrays to be summed.
    dtype : numpy.dtype, optional
        The type of the yielded array and of the accumulator in which the elements 
        are summed. The dtype of a is used by default unless a has an integer dtype 
        of less precision than the default platform integer. In that case, if a is 
        signed then the platform integer is used while if a is unsigned then an 
        unsigned integer of the same precision as the platform integer is used.
    
    Yields
    ------
    online_sum : ndarray
    """
    ignored_nans = map(np.nan_to_num, arrays)
    yield from isum(ignored_nans, dtype = dtype)

def iprod(arrays, dtype = None):
    """ 
    Streaming product of array elements.

    Parameters
    ----------
    arrays : iterable
        Arrays to be multiplied.
    dtype : numpy.dtype, optional
        The type of the yielded array and of the accumulator in which the elements 
        are summed. The dtype of a is used by default unless a has an integer dtype 
        of less precision than the default platform integer. In that case, if a is 
        signed then the platform integer is used while if a is unsigned then an 
        unsigned integer of the same precision as the platform integer is used.
    
    Yields
    ------
    online_prod : ndarray
    """
    arrays = iter(arrays)

    first = next(arrays)
    if dtype is None:
        dtype = first.dtype
    
    accumulator = first.astype(dtype, copy = True)
    for array in arrays:
        accumulator *= array.astype(dtype, copy = False)
        yield accumulator

def inanprod(arrays, dtype = None):
    """ 
    Streaming product of array elements. NaNs are ignored (i.e. treated as one).

    Parameters
    ----------
    arrays : iterable
        Arrays to be multiplied.
    dtype : numpy.dtype, optional
        The type of the yielded array and of the accumulator in which the elements 
        are summed. The dtype of a is used by default unless a has an integer dtype 
        of less precision than the default platform integer. In that case, if a is 
        signed then the platform integer is used while if a is unsigned then an 
        unsigned integer of the same precision as the platform integer is used.
    
    Yields
    ------
    online_prod : ndarray
    """
    ignored_nans = map(partial(_nan_to_num, fill = 1.0), arrays)
    yield from iprod(ignored_nans, dtype = dtype)