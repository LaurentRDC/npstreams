# -*- coding: utf-8 -*-
"""
Numerics Functions
------------------
"""
from itertools import chain
import numpy as np
from functools import partial
from . import preduce, last, chunked, stream_reduce

def isum(arrays, axis = -1, dtype = None, ignore_nan = False):
    """ 
    Streaming sum of array elements.

    Parameters
    ----------
    arrays : iterable
        Arrays to be summed. 
    axis : int or None, optional
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
    npfunc = np.sum
    if ignore_nan:
        npfunc = np.nansum
    
    yield from stream_reduce(arrays, npfunc = npfunc, axis = axis, dtype = dtype)

def inansum(arrays, axis = -1, dtype = None):
    """ 
    Streaming sum of array elements. NaNs are ignored (i.e. treated as zero).

    Parameters
    ----------
    arrays : iterable
        Arrays to be summed.
    axis : int or None, optional
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
    axis : int or None, optional
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
    # TODO: parallelize
    kwargs = {'ignore_nan': ignore_nan, 'dtype': dtype, 'axis': axis}
    return last(isum(arrays, **kwargs))

def iprod(arrays, axis = -1, dtype = None, ignore_nan = False):
    """ 
    Streaming product of array elements.

    Parameters
    ----------
    arrays : iterable
        Arrays to be multiplied.
    axis : int or None, optional
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
    npfunc = np.prod
    if ignore_nan:
        npfunc = np.nanprod
    
    yield from stream_reduce(arrays, npfunc = npfunc, axis = axis, dtype = dtype)

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
    axis : int or None, optional
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
    # TODO: parallelize using preduce
    kwargs = {'ignore_nan': ignore_nan, 'dtype': dtype, 'axis': axis}
    return last(iprod(arrays, **kwargs))

def inanprod(arrays, axis = -1, dtype = None):
    """ 
    Streaming product of array elements. NaNs are ignored (i.e. treated as one).

    Parameters
    ----------
    arrays : iterable
        Arrays to be multiplied.
    axis : int or None, optional
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

def isub(arrays, axis = -1, dtype = None):
    """
    Subtract elements in a reduction fashion. Equivalent to ``numpy.subtract.reduce`` on a dense array.

    Parameters
    ----------
    arrays : iterable
        Arrays to be multiplied.
    axis : int, optional
        Reduction axis. Since subtraction is not reorderable (unlike a sum, for example),
        `axis` must be specified as an int; full reduction (``axis = None``) will raise an exception. 
        Default is to subtract the arrays in the stream as if they had been stacked along a new axis, 
        then subtract along this new axis. If None, arrays are flattened before subtraction. 
        If `axis` is an int larger that the number of dimensions in the arrays of the stream, 
        arrays are subtracted along the new axis.
    dtype : numpy.dtype, optional
        The type of the yielded array and of the accumulator in which the elements 
        are combined. The dtype of a is used by default unless a has an integer dtype 
        of less precision than the default platform integer. In that case, if a is 
        signed then the platform integer is used while if a is unsigned then an 
        unsigned integer of the same precision as the platform integer is used.
    
    Yields
    ------
    online_sub : ndarray

    Raises
    ------
    ValueError
        If `axis` is None. Since subtraction is not reorderable (unlike a sum, for example),
        `axis` must be specified as an int.
    """
    if axis is None:
        raise ValueError('Subtraction is not a reorderable operation, and \
                          therefore a specific axis must be give.')
    yield from stream_reduce(arrays, npfunc = np.subtract.reduce, axis = axis, dtype = dtype)