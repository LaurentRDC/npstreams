# -*- coding: utf-8 -*-
"""
Numerics Functions
------------------
"""
import numpy as np
from functools import partial
from . import _nan_to_num, preduce, last, chunked

def isum(arrays, dtype = None, ignore_nan = False):
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
    ignore_nan : bool, optional
        If True, NaNs are ignored. Default is propagation of NaNs.
    
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
        if ignore_nan:  # TODO: also check if array of floats or complex
            array = np.nan_to_num(array)
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
    yield from isum(arrays, dtype = dtype, ignore_nan = True)

# Can't pickle local functions, so it must be defined here
# for use in psum
def _sumf(array1, array2, **kwargs):
    return last(isum([array1, array2], **kwargs))

def psum(arrays, dtype = None, ignore_nan = False, processes = 1):
    """ 
    Parallel sum of array elements.

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
    ignore_nan : bool, optional
        If True, NaNs are ignored. Default is propagation of NaNs.
    processes : int or None, optional
        Number of processes to use. If `None`, maximal number of processes
        is used. Default is one.
    
    Returns
    -------
    sum : ndarray
    """
    return preduce(_sumf, arrays, 
                   kwargs = {'ignore_nan': ignore_nan, 'dtype': dtype},
                   processes = processes)

def iprod(arrays, dtype = None, ignore_nan = False):
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
    ignore_nan : bool, optional
        If True, NaNs are ignored. Default is propagation of NaNs.
    
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
        if ignore_nan:
            array = _nan_to_num(array, 1)
        accumulator *= array.astype(dtype, copy = False)
        yield accumulator

# Can't pickle local functions, so it must be defined here
# for use in pprod
def _prodf(array1, array2, **kwargs):
    return last(iprod([array1, array2], **kwargs))

def pprod(arrays, dtype = None, ignore_nan = False, processes = 1):
    """ 
    Parallel product of array elements.

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
    ignore_nan : bool, optional
        If True, NaNs are ignored. Default is propagation of NaNs.
    processes : int or None, optional
        Number of processes to use. If `None`, maximal number of processes
        is used. Default is one.
    
    Returns
    -------
    prod : ndarray
    """
    return preduce(_prodf, arrays, 
                   kwargs = {'ignore_nan': ignore_nan, 'dtype': dtype},
                   processes = processes)


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
    yield from iprod(arrays, dtype = dtype, ignore_nan = True)