# -*- coding: utf-8 -*-
"""
Numerics Functions
------------------
"""
import numpy as np
from functools import partial
from . import preduce, last, chunked, stream_ufunc, array_stream

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
    # TODO: don't map nan_to_num if arrays aren't floats
    if ignore_nan:
        arrays = map(np.nan_to_num, arrays)
    
    yield from stream_ufunc(arrays, ufunc = np.add, axis = axis, dtype = dtype)

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
    if ignore_nan:
        def _nan_to_num(array): #numpy nan_to_num() replaces all NaNs with zeros
            array = np.array(array, copy = True)
            array[np.isnan(array)] = 1
            return array
        arrays = map(_nan_to_num, arrays)

    yield from stream_ufunc(arrays, ufunc = np.multiply, axis = axis, dtype = dtype)

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
    yield from stream_ufunc(arrays, ufunc = np.subtract, axis = axis, dtype = dtype)

def iall(arrays, axis = -1):
    """ 
    Test whether all array elements along a given axis evaluate to True 
    
    Parameters
    ----------
    arrays : iterable
        Arrays to be reduced.
    axis : int or None, optional
        Axis along which a logical AND reduction is performed. The default
        is to perform a logical AND along the 'stream axis', as if all arrays in ``array``
        were stacked along a new dimension. If ``axis = None``, arrays in ``arrays`` are flattened
        before reduction.

    Yields
    ------
    all : ndarray, dtype bool 
    """
    yield from stream_ufunc(arrays, ufunc = np.logical_and, axis = axis)

def iany(arrays, axis = -1):
    """ 
    Test whether any array elements along a given axis evaluate to True.
    
    Parameters
    ----------
    arrays : iterable
        Arrays to be reduced.
    axis : int or None, optional
        Axis along which a logical OR reduction is performed. The default
        is to perform a logical AND along the 'stream axis', as if all arrays in ``array``
        were stacked along a new dimension. If ``axis = None``, arrays in ``arrays`` are flattened
        before reduction.

    Yields
    ------
    any : ndarray, dtype bool 
    """
    yield from stream_ufunc(arrays, ufunc = np.logical_or, axis = axis)