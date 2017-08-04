# -*- coding: utf-8 -*-
"""
General stream reduction
------------------------
"""
import numpy as np
from functools import partial
from itertools import chain
from . import peek, array_stream

# TODO: initializer
# TODO: keepdims?
@array_stream
def stream_reduce(arrays, npfunc, axis = -1, **kwargs):
    """
    Reduction operation for a stream of arrays. Applies a reduction function
    progressively. 
    
    Parameters
    ----------
    arrays : iterable
        Arrays to be reduced.
    npfunc : callable
        NumPy reduction function. This function must support the `axis` parameter.
    axis : int or None, optional
        Reduction axis. Default is to reduce the arrays in the stream as if 
        they had been stacked along a new axis, then reduce along this new axis.
        If None, arrays are flattened before reduction. If `axis` is an int larger that
        the number of dimensions in the arrays of the stream, arrays are reduced
        along the new axis. Note that not all of NumPy reduction functions support 
        ``axis = None``, e.g. ``numpy.subtract.reduce``.
    kwargs
        Keyword arguments are passed to ``npfunc``.
    
    Yields 
    ------
    reduced : ndarray or scalar
    """
    if axis is None:
        yield from _stream_reduce_all_axes(arrays, npfunc, **kwargs)
        return

    if axis == -1:
        yield from _stream_reduce_new_axis(arrays, npfunc, **kwargs)
        return

    first, arrays = peek(arrays)
    
    if axis >= first.ndim:
        yield from stream_reduce(arrays, npfunc, axis = -1, **kwargs)
        return

    yield from _stream_reduce_existing_axis(arrays, axis = axis, npfunc = npfunc, **kwargs)

def _stream_reduce_new_axis(arrays, npfunc, **kwargs):
    """
    Reduction operation for arrays, in the direction of a new axis (i.e. stacking).
    
    Parameters
    ----------
    arrays : iterable
        Arrays to be reduced.
    npfunc : callable
        NumPy reduction function. This function must support the `axis`
        parameter, e.g. numpy.sum.
    kwargs
        Keyword arguments are passed to ``npfunc``.
    
    Yields 
    ------
    reduced : ndarray
    """
    arrays = iter(arrays)
    first = next(arrays)

    dtype = kwargs.get('dtype')
    if dtype is None:
        dtype = first.dtype
    
    axis_reduce = partial(npfunc, axis = first.ndim, **kwargs)
                
    accumulator = np.array(first, copy = True).astype(dtype)
    yield accumulator
    
    for array in arrays:
        accumulator = axis_reduce(np.stack([accumulator, array], axis = -1), out = accumulator)
        yield accumulator

def _stream_reduce_existing_axis(arrays, axis, npfunc, **kwargs):
    """
    Reduction operation for arrays, in the direction of a new axis (i.e. stacking).
    
    Parameters
    ----------
    arrays : iterable
        Arrays to be reduced.
    axis : int
        Axis along which a reduction is performed. 
    npfunc : callable
        NumPy reduction function. This function must support the `axis`
        parameter, e.g. numpy.sum.
    kwargs
        Keyword arguments are passed to ``npfunc``.

    Yields 
    ------
    reduced : ndarray
    """
    arrays = iter(arrays)
    first = next(arrays)

    if axis not in range(first.ndim):
        raise ValueError('Axis {} not supported on arrays of shape {}.'.format(axis, first.shape))
    
    dtype = kwargs.get('dtype')
    if dtype is None:
        dtype = first.dtype
    
    axis_reduce = partial(npfunc, axis = axis, **kwargs)

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
    
def _stream_reduce_all_axes(arrays, npfunc, **kwargs):
    """
    Reduction operation for arrays, over all axes.
    
    Parameters
    ----------
    arrays : iterable
        Arrays to be reduced.
    npfunc : callable
        NumPy reduction function. This function must support the `axis` 
        parameter, e.g. `numpy.sum`.
    kwargs
        Keyword arguments are passed to ``npfunc``.

    Yields 
    ------
    reduced : scalar
    """
    arrays = iter(arrays)
    first = next(arrays)

    axis_reduce = partial(npfunc, axis = None, **kwargs)

    accumulator = axis_reduce(first)
    yield accumulator
    
    for array in arrays:
        accumulator = axis_reduce([accumulator, axis_reduce(array)])
        yield accumulator