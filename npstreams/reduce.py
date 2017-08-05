# -*- coding: utf-8 -*-
"""
General stream reduction
------------------------
"""
import numpy as np
from functools import partial, wraps
from itertools import chain
from . import peek, array_stream

# Priming a generator allows the execution of error-checking
# code immediatly. See ireduce_ufunc for an example
def primed(gen):
    """ Primes a generator. Useful in cases where there are preliminary checks
    when creating the generator """
    @wraps(gen)
    def primed_gen(*args, **kwargs):
        generator = gen(*args, **kwargs)
        next(generator)
        return generator
    return primed_gen

@primed
@array_stream
def ireduce_ufunc(arrays, ufunc, axis = -1, dtype = None, **kwargs):
    """
    Create a streaming reduction function from a binary NumPy ufunc.

    Note that while all ufuncs have a ``reduce`` method, not all of them are useful.
    
    Parameters
    ----------
    arrays : iterable
        Arrays to be reduced.
    ufunc : numpy.ufunc
        Binary universal function. Must have a signature of the form ufunc(x1, x2, ...)
    axis : int or None, optional
        Reduction axis. Default is to reduce the arrays in the stream as if 
        they had been stacked along a new axis, then reduce along this new axis.
        If None, arrays are flattened before reduction. If `axis` is an int larger that
        the number of dimensions in the arrays of the stream, arrays are reduced
        along the new axis. Note that not all of NumPy Ufuncs support 
        ``axis = None``, e.g. ``numpy.subtract``.
    dtype : numpy.dtype or None, optional
        Overrides the dtype of the calculation and output arrays.
    kwargs
        Keyword arguments are passed to ``ufunc``. Note that some valid ufunc keyword arguments
        (e.g. ``keepdims``) are not valid for all streaming functions.
    
    Yields 
    ------
    reduced : ndarray or scalar

    Raises
    ------
    TypeError : if ``ufunc`` is not a binary universal function.
    """
    kwargs.update({'dtype': dtype, 'axis': axis})

    try:
        assert isinstance(ufunc, np.ufunc)
        ufunc.reduce([1,2], axis = 0)
    except (ValueError, AssertionError):
        raise TypeError('Only binary ufuncs are supported, and {} is not one of them'.format(ufunc.__name__))
    
    # Since ireduce_ufunc is primed, we need to wait here
    yield

    if kwargs['axis'] is None:
        yield from _ireduce_ufunc_all_axes(arrays, ufunc, **kwargs)
        return

    if kwargs['axis'] == -1:
        yield from _ireduce_ufunc_new_axis(arrays, ufunc, **kwargs)
        return

    first, arrays = peek(arrays)
    
    if kwargs['axis'] >= first.ndim:
        kwargs['axis'] = -1
        yield from ireduce_ufunc(arrays, ufunc, **kwargs)
        return

    yield from _ireduce_ufunc_existing_axis(arrays, ufunc, **kwargs)

def _ireduce_ufunc_new_axis(arrays, ufunc, **kwargs):
    """
    Reduction operation for arrays, in the direction of a new axis (i.e. stacking).
    
    Parameters
    ----------
    arrays : iterable
        Arrays to be reduced.
    ufunc : numpy.ufunc
        Binary universal function. Must have a signature of the form ufunc(x1, x2, ...)
    kwargs
        Keyword arguments are passed to ``ufunc``
    
    Yields 
    ------
    reduced : ndarray
    """
    arrays = iter(arrays)
    first = next(arrays)
    
    kwargs['axis'] = first.ndim

    axis_reduce = partial(ufunc.reduce, **kwargs)
                
    dtype = kwargs.get('dtype', None)
    if dtype is None:
        dtype = first.dtype
    accumulator = np.array(first, copy = True).astype(dtype)
    yield accumulator
    
    for array in arrays:
        accumulator = axis_reduce(np.stack([accumulator, array], axis = -1), out = accumulator)
        yield accumulator

def _ireduce_ufunc_existing_axis(arrays, ufunc, **kwargs):
    """
    Reduction operation for arrays, in the direction of a new axis (i.e. stacking).
    
    Parameters
    ----------
    arrays : iterable
        Arrays to be reduced.
    ufunc : numpy.ufunc
        Binary universal function. Must have a signature of the form ufunc(x1, x2, ...)
    kwargs
        Keyword arguments are passed to ``ufunc``

    Yields 
    ------
    reduced : ndarray
    """
    arrays = iter(arrays)
    first = next(arrays)

    if kwargs['axis'] not in range(first.ndim):
        raise ValueError('Axis {} not supported on arrays of shape {}.'.format(axis, first.shape))
    
    dtype = kwargs.get('dtype')
    if dtype is None:
        dtype = first.dtype
    
    axis_reduce = partial(ufunc.reduce, **kwargs)

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
    
def _ireduce_ufunc_all_axes(arrays, ufunc, **kwargs):
    """
    Reduction operation for arrays, over all axes.
    
    Parameters
    ----------
    arrays : iterable
        Arrays to be reduced.
    ufunc : numpy.ufunc
        Binary universal function. Must have a signature of the form ufunc(x1, x2, ...)
    kwargs
        Keyword arguments are passed to ``ufunc``

    Yields 
    ------
    reduced : scalar
    """
    arrays = iter(arrays)
    first = next(arrays)

    kwargs['axis'] = None
    axis_reduce = partial(ufunc.reduce, **kwargs)

    accumulator = axis_reduce(first)
    yield accumulator
    
    for array in arrays:
        accumulator = axis_reduce([accumulator, axis_reduce(array)])
        yield accumulator