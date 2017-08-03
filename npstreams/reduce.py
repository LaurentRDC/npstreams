# -*- coding: utf-8 -*-
"""
Numerics Functions
------------------
"""
import numpy as np
from functools import partial
from itertools import chain
from . import peek

# TODO: initializer
# TODO: keepdims
def stream_reduce(arrays, npfunc, axis = -1, dtype = None):
    """
    Reduction operation for a stream of arrays. Applies a reduction function
    progressively. 
    
    Parameters
    ----------
    arrays : iterable
        Arrays to be reduced.
    npfunc : callable
        NumPy reduction function. This function must support the `axis` parameter.
    axis : int, optional
        Reduction axis. Default is to reduce the arrays in the stream as if 
        they had been stacked along a new axis, then reduce along this new axis.
        If None, arrays are flattened before reduction. If `axis` is an int larger that
        the number of dimensions in the arrays of the stream, arrays are reduced
        along the new axis.
    dtype : numpy.dtype, optional
        The type of the yielded array and of the accumulator in which the elements 
        are reduced. The dtype of a is used by default unless a has an integer dtype 
        of less precision than the default platform integer. In that case, if a is 
        signed then the platform integer is used while if a is unsigned then an 
        unsigned integer of the same precision as the platform integer is used.
    
    Yields 
    ------
    reduced : ndarray or scalar
    """
    if axis is None:
        yield from _stream_reduce_all_axes(arrays, npfunc, dtype)
        return

    if axis == -1:
        yield from _stream_reduce_new_axis(arrays, npfunc, dtype)
        return

    first, arrays = peek(arrays)
    
    if axis >= first.ndim:
        yield from stream_reduce(arrays, npfunc, axis = -1, dtype = dtype)
        return

    yield from _stream_reduce_existing_axis(arrays, axis = axis, npfunc = npfunc, dtype = dtype)

def _stream_reduce_new_axis(arrays, npfunc, dtype):
    """
    Reduction operation for arrays, in the direction of a new axis (i.e. stacking).
    
    Parameters
    ----------
    arrays : iterable
        Arrays to be reduced.
    npfunc : callable
        NumPy reduction function. This function must support the `axis` and `dtype` 
        parameters, e.g. numpy.sum.
    dtype : numpy.dtype, optional
        The type of the yielded array and of the accumulator in which the elements 
        are reduced. The dtype of a is used by default unless a has an integer dtype 
        of less precision than the default platform integer. In that case, if a is 
        signed then the platform integer is used while if a is unsigned then an 
        unsigned integer of the same precision as the platform integer is used.

    Yields 
    ------
    reduced : ndarray
    """
    arrays = iter(arrays)
    first = next(arrays)

    if dtype is None:
        dtype = first.dtype
    
    axis_reduce = partial(npfunc, axis = first.ndim, dtype = dtype)
                
    accumulator = np.array(first, copy = True).astype(dtype)
    yield accumulator
    
    for array in arrays:
        accumulator = axis_reduce(np.stack([accumulator, array], axis = -1), out = accumulator)
        yield accumulator

def _stream_reduce_existing_axis(arrays, axis, npfunc, dtype = None):
    """
    Reduction operation for arrays, in the direction of a new axis (i.e. stacking).
    
    Parameters
    ----------
    arrays : iterable
        Arrays to be reduced.
    axis : int
        Axis along which a reduction is performed. 
    npfunc : callable
        NumPy reduction function. This function must support the `axis` and `dtype` 
        parameters, e.g. numpy.sum.
    dtype : numpy.dtype, optional
        The type of the yielded array and of the accumulator in which the elements 
        are reduced. The dtype of a is used by default unless a has an integer dtype 
        of less precision than the default platform integer. In that case, if a is 
        signed then the platform integer is used while if a is unsigned then an 
        unsigned integer of the same precision as the platform integer is used.

    Yields 
    ------
    reduced : ndarray
    """
    arrays = iter(arrays)
    first = next(arrays)

    if axis not in range(first.ndim):
        raise ValueError('Axis {} not supported on arrays of shape {}.'.format(axis, first.shape))
    
    if dtype is None:
        dtype = first.dtype
    
    axis_reduce = partial(npfunc, axis = axis, dtype = dtype)

    accumulator = np.atleast_1d(axis_reduce(first))
    yield accumulator

    # On the first pass of the following loop, accumulator is missing a dimensions
    # therefore, the stacking function cannot be 'concatenate'
    second = next(arrays)
    accumulator = np.stack([accumulator, np.atleast_1d(axis_reduce(second))], axis = -1)
    yield accumulator

    # On the second pass, the new dimensions exists, and thus we switch to
    # using concatenate.
    for array in arrays:
        reduced = np.expand_dims(np.atleast_1d(axis_reduce(array)), axis = accumulator.ndim - 1)
        accumulator = np.concatenate([accumulator, reduced], axis = accumulator.ndim - 1)
        yield accumulator
    
def _stream_reduce_all_axes(arrays, npfunc, dtype = None):
    """
    Reduction operation for arrays, over all axes.
    
    Parameters
    ----------
    arrays : iterable
        Arrays to be reduced.
    npfunc : callable
        NumPy reduction function. This function must support the `axis` and `dtype` 
        parameters, e.g. `numpy.sum`.
    dtype : numpy.dtype, optional
        The type of the yielded array and of the accumulator in which the elements 
        are reduced. The dtype of a is used by default unless a has an integer dtype 
        of less precision than the default platform integer. In that case, if a is 
        signed then the platform integer is used while if a is unsigned then an 
        unsigned integer of the same precision as the platform integer is used.

    Yields 
    ------
    reduced : scalar
    """
    arrays = iter(arrays)
    first = next(arrays)

    axis_reduce = partial(npfunc, axis = None, dtype = dtype)

    accumulator = axis_reduce(first)
    yield accumulator
    
    for array in arrays:
        accumulator = axis_reduce([accumulator, axis_reduce(array)])
        yield accumulator

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
    # Since stream_reduce will pass a 'dtype' parameter, we must ignore it
    # TODO: this is clunky
    def all_ignore_kwargs(*args, **kwargs):
        kwargs.pop('dtype')
        return np.all(*args, **kwargs)
    yield from stream_reduce(arrays, npfunc = all_ignore_kwargs, axis = axis)

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
    # Since stream_reduce will pass a 'dtype' parameter, we must ignore it
    # TODO: this is clunky
    def any_ignore_kwargs(*args, **kwargs):
        kwargs.pop('dtype')
        return np.any(*args, **kwargs)
    yield from stream_reduce(arrays, npfunc = any_ignore_kwargs, axis = axis)
