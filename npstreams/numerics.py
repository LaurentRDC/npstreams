# -*- coding: utf-8 -*-
"""
Numerics Functions
------------------
"""
import numpy as np
from functools import partial
from . import _nan_to_num, preduce, last, chunked

def isum(arrays, axis = -1, dtype = None, ignore_nan = False):
    """ 
    Streaming sum of array elements.

    Parameters
    ----------
    arrays : iterable
        Arrays to be summed.
    axis : int, optional
        Reduction axis. Default is to sum the arrays in the stream as if 
        they had been stacked along a new axis, then sum along this new axis.
        If None, arrays are flattened before summing. If `axis` is an int larger that
        the number of dimensions in the arrays of the stream, arrays are summed
        along the new axis.
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
    
    if axis is not None:
        if axis > first.ndim:
            axis = -1

    # Before the array is accumulated, it might be reduced based on axis
    # parameter or dtype
    axis_reduce = lambda x: x.astype(dtype, copy = False)
    if axis != -1:
        axis_reduce = partial(np.sum, axis = axis, dtype = dtype)
    
    if ignore_nan:
        first = np.nan_to_num(first)
    
    accumulator = axis_reduce(first)
    for array in arrays:
        if ignore_nan:  # TODO: also check if array of floats or complex
            array = np.nan_to_num(array)
        accumulator += axis_reduce(array)
        yield accumulator

def inansum(arrays, axis = -1, dtype = None):
    """ 
    Streaming sum of array elements. NaNs are ignored (i.e. treated as zero).

    Parameters
    ----------
    arrays : iterable
        Arrays to be summed.
    axis : int, optional
        Reduction axis. Default is to sum the arrays in the stream as if 
        they had been stacked along a new axis, then sum along this new axis.
        If None, arrays are flattened before summing. If `axis` is an int larger that
        the number of dimensions in the arrays of the stream, arrays are summed
        along the new axis.
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
    yield from isum(arrays, axis = axis, dtype = dtype, ignore_nan = True)

# Can't pickle local functions, so it must be defined here
# for use in psum
def _sumf(array1, array2, **kwargs):
    return last(isum([array1, array2], **kwargs))

def psum(arrays, axis = -1, dtype = None, ignore_nan = False, processes = 1):
    """ 
    Parallel sum of array elements.

    Parameters
    ----------
    arrays : iterable
        Arrays to be summed.
    axis : int, optional
        Reduction axis. Default is to sum the arrays in the stream as if 
        they had been stacked along a new axis, then sum along this new axis.
        If None, arrays are flattened before summing. If `axis` is an int larger that
        the number of dimensions in the arrays of the stream, arrays are summed
        along the new axis.
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
    return preduce(_sumf, arrays, processes = processes,
                   kwargs = {'ignore_nan': ignore_nan, 
                             'dtype': dtype, 
                             'axis': axis})

def iprod(arrays, axis = -1, dtype = None, ignore_nan = False):
    """ 
    Streaming product of array elements.

    Parameters
    ----------
    arrays : iterable
        Arrays to be multiplied.
    axis : int, optional
        Reduction axis. Default is to multiply the arrays in the stream as if 
        they had been stacked along a new axis, then multiply along this new axis.
        If None, arrays are flattened before multiplication. If `axis` is an int larger that
        the number of dimensions in the arrays of the stream, arrays are multiplied
        along the new axis.
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
    
    if axis is not None:
        if axis > first.ndim:
            axis = -1

    # Before the array is accumulated, it might be reduced based on axis
    # parameter or dtype
    axis_reduce = lambda x: x.astype(dtype, copy = False)
    if axis != -1:
        axis_reduce = partial(np.prod, axis = axis, dtype = dtype)
    
    if ignore_nan:
        first = _nan_to_num(first, 1)
    
    accumulator = axis_reduce(first)
    for array in arrays:
        if ignore_nan:  # TODO: also check if array of floats or complex
            array = _nan_to_num(array, 1)
        accumulator *= axis_reduce(array)
        yield accumulator

# Can't pickle local functions, so it must be defined here
# for use in pprod
def _prodf(array1, array2, **kwargs):
    return last(iprod([array1, array2], **kwargs))

def pprod(arrays, axis = -1, dtype = None, ignore_nan = False, processes = 1):
    """ 
    Parallel product of array elements.

    Parameters
    ----------
    arrays : iterable
        Arrays to be multiplied.
    axis : int, optional
        Reduction axis. Default is to multiply the arrays in the stream as if 
        they had been stacked along a new axis, then multiply along this new axis.
        If None, arrays are flattened before multiplication. If `axis` is an int larger that
        the number of dimensions in the arrays of the stream, arrays are multiplied
        along the new axis.
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
    return preduce(_prodf, arrays, processes = processes,
                   kwargs = {'ignore_nan': ignore_nan, 
                             'dtype': dtype,
                             'axis': axis})


def inanprod(arrays, axis = -1, dtype = None):
    """ 
    Streaming product of array elements. NaNs are ignored (i.e. treated as one).

    Parameters
    ----------
    arrays : iterable
        Arrays to be multiplied.
    axis : int, optional
        Reduction axis. Default is to multiply the arrays in the stream as if 
        they had been stacked along a new axis, then multiply along this new axis.
        If None, arrays are flattened before multiplication. If `axis` is an int larger that
        the number of dimensions in the arrays of the stream, arrays are multiplied
        along the new axis.
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
    yield from iprod(arrays, axis = axis, dtype = dtype, ignore_nan = True)