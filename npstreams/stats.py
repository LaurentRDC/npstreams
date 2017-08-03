# -*- coding: utf-8 -*-
"""
Statistical functions
---------------------
"""
from functools import partial
from itertools import repeat, tee, chain
from cytoolz import peek
import numpy as np
from math import sqrt
from .numerics import isum
from . import _nan_to_num

def _atleast_array(arg, arr):
    """ Make sure that if inputs are float or int, they are cast to array of shape `shape` """
    if isinstance(arg, (float, int)):
        arg = np.full(shape = arr.shape, fill_value = arg, dtype = arr.dtype)
    else:
        arg = np.asarray(arg)
    return arg

def _weighted(arrays, weights, ignore_nan = False):
    for array, weight in zip(arrays, weights):
        if ignore_nan:
            weight[np.isnan(array)] = 0
            array = np.nan_to_num(array)
        yield array * weight

def iaverage(arrays, axis = -1, weights = None, ignore_nan = False):
    """ 
    Streaming (weighted) average of arrays.

    Parameters
    ----------
    arrays : iterable of ndarrays
        Arrays to be averaged. This iterable can also a generator.
    axis : int, optional
        Reduction axis. Default is to average the arrays in the stream as if 
        they had been stacked along a new axis, then average along this new axis.
        If None, arrays are flattened before averaging. If `axis` is an int larger that
        the number of dimensions in the arrays of the stream, arrays are averaged
        along the new axis.
    weights : iterable of ndarray, iterable of floats, or None, optional
        Iterable of weights associated with the values in each item of `images`. 
        Each value in an element of `images` contributes to the average 
        according to its associated weight. The weights array can either be a float
        or an array of the same shape as any element of `images`. If weights=None, 
        then all data in each element of `images` are assumed to have a weight equal to one.
    ignore_nan : bool, optional
        If True, NaNs are set to zero weight. Default is propagation of NaNs.
    
    Yields
    ------
    avg: `~numpy.ndarray`
        Weighted average. 
    
    See Also
    --------
    imean : streaming array mean.
    numpy.average : (weighted) average for dense arrays
    """
    first, arrays = peek(arrays)
    
    # We make sure that weights is always an array
    # This simplifies the handling of NaNs.
    if weights is None:
        weights = repeat(1)
    weights = map(partial(np.broadcast_to, shape = first.shape), weights)

    # Need to know which array has NaNs, and modify the weights stream accordingly
    if ignore_nan:
        arrays, arrays2 = tee(arrays)
        weights = map(lambda arr, wgt: np.logical_not(np.isnan(arr)) * wgt, arrays2, weights)
        arrays = map(np.nan_to_num, arrays)

    weights1, weights2 = tee(weights)

    sum_of_weights = isum(weights1, axis = axis)
    weighted_arrays = map(lambda arr, wgt: arr * wgt, arrays, weights2)
    weighted_sum = isum(weighted_arrays, axis = axis, ignore_nan = ignore_nan)
    
    yield from map(lambda arr, wgt: arr/wgt, weighted_sum, sum_of_weights)

def imean(arrays, axis = -1, ignore_nan = False):
    """ 
    Streaming mean of arrays. Equivalent to `iaverage(arrays, weights = None)`.

    Parameters
    ----------
    arrays : iterable of ndarrays
        Arrays to be averaged. This iterable can also a generator.
    axis : int, optional
        Reduction axis. Default is to average the arrays in the stream as if 
        they had been stacked along a new axis, then average along this new axis.
        If None, arrays are flattened before averaging. If `axis` is an int larger that
        the number of dimensions in the arrays of the stream, arrays are averaged
        along the new axis.
    ignore_nan : bool, optional
        If True, NaNs are set to zero weight. Default is propagation of NaNs.
    
    Yields
    ------
    mean: `~numpy.ndarray`
        Online mean array.
    """
    yield from iaverage(arrays, axis = axis, weights = None, ignore_nan = ignore_nan)

def inanmean(arrays, axis = -1):
    """ 
    Streaming mean of arrays, ignoring NaNs. Equivalent to `imean(ignore_nan = True)`.

    Parameters
    ----------
    arrays : iterable of ndarrays
        Arrays to be averaged. This iterable can also a generator.
    axis : int, optional
        Reduction axis. Default is to average the arrays in the stream as if 
        they had been stacked along a new axis, then average along this new axis.
        If None, arrays are flattened before averaging. If `axis` is an int larger that
        the number of dimensions in the arrays of the stream, arrays are averaged
        along the new axis.
    
    Yields
    ------
    mean: `~numpy.ndarray`
        Online mean array.
    """
    yield from imean(arrays, axis = axis, ignore_nan = True)

def ivar(arrays, axis = -1, ddof = 0, weights = None, ignore_nan = False):
    """ 
    Streaming variance of arrays. Weights are also supported.
    
    Parameters
    ----------
    arrays : iterable of ndarrays
        Arrays to be averaged. This iterable can also a generator.
    axis : int, optional
        Reduction axis. Default is to combine the arrays in the stream as if 
        they had been stacked along a new axis, then compute the variance along this new axis.
        If None, arrays are flattened. If `axis` is an int larger that
        the number of dimensions in the arrays of the stream, variance is computed
        along the new axis.
    ddof : int, optional
        Means Delta Degrees of Freedom.  The divisor used in calculations
        is ``N - ddof``, where ``N`` represents the number of elements.
        By default `ddof` is one.
    weights : iterable of ndarray, iterable of floats, or None, optional
        Iterable of weights associated with the values in each item of `arrays`. 
        Each value in an element of `arrays` contributes to the variance 
        according to its associated weight. The weights array can either be a float
        or an array of the same shape as any element of `arrays`. If weights=None, 
        then all data in each element of `arrays` are assumed to have a weight equal to one.
    ignore_nan : bool, optional
        If True, NaNs are set to zero weight. Default is propagation of NaNs.
    
    Yields
    ------
    var: `~numpy.ndarray`
        Variance. 
    
    See Also
    --------
    numpy.var : variance calculation for dense arrays. Weights are not supported.
    
    References
    ----------
    .. [#] D. H. D. West, Updating the mean and variance estimates: an improved method.
        Communications of the ACM Vol. 22, Issue 9, pp. 532 - 535 (1979)
    """
    arrays = iter(arrays)
    first = next(arrays)

    # We make sure that weights is always an array
    # This simplifies the handling of NaNs.
    if weights is None:
        weights = repeat(1)
    weights = map(partial(_atleast_array, arr = first), iter(weights))
    first_weight = next(weights)

    axis_reduce = lambda x: x
    if axis != -1:
        axis_reduce = partial(np.sum, axis = axis)
        if ignore_nan:
            axis_reduce = partial(np.nansum, axis = axis)
    
    first = axis_reduce(first)
    sum_of_weights = np.array(axis_reduce(first_weight), copy = True)
    if ignore_nan:
        sum_of_weights[np.isnan(first)] = 0
        first = np.nan_to_num(first)

    old_mean = np.array(first, copy = True)
    new_mean = np.array(first, copy = True)
    old_S = np.zeros_like(first, dtype = np.float)
    new_S = np.zeros_like(first, dtype = np.float)
    yield np.zeros_like(first)
    
    for array, weight in zip(arrays, weights):
        array = axis_reduce(array)
        weight = axis_reduce(weight)

        if ignore_nan:
            weight[np.isnan(array)] = 0
            array = np.nan_to_num(array)

        sum_of_weights += weight

        _sub = weight * (array - old_mean)
        new_mean = old_mean + _sub/sum_of_weights
        new_S = old_S + _sub*(array - new_mean)

        # In case there hasn't been enough measurements yet,
        # yield zeros.
        if np.any(sum_of_weights - ddof <= 0):
            yield np.zeros_like(first)
        else:
            yield new_S/(sum_of_weights - ddof) # variance = S / k-1, sem = std / sqrt(k)    

        old_mean = new_mean
        old_S = new_S

def inanvar(arrays, axis = -1, ddof = 0, weights = None):
    """ 
    Streaming variance of arrays. Weights are also supported. NaNs are ignored.
    Equivalent to `ivarignore_nan = True)`.
    
    Parameters
    ----------
    arrays : iterable of ndarrays
        Arrays to be averaged. This iterable can also a generator.
    axis : int, optional
        Reduction axis. Default is to combine the arrays in the stream as if 
        they had been stacked along a new axis, then compute the variance along this new axis.
        If None, arrays are flattened. If `axis` is an int larger that
        the number of dimensions in the arrays of the stream, variance is computed
        along the new axis.
    ddof : int, optional
        Means Delta Degrees of Freedom.  The divisor used in calculations
        is ``N - ddof``, where ``N`` represents the number of elements.
        By default `ddof` is one.
    weights : iterable of ndarray, iterable of floats, or None, optional
        Iterable of weights associated with the values in each item of `arrays`. 
        Each value in an element of `arrays` contributes to the variance 
        according to its associated weight. The weights array can either be a float
        or an array of the same shape as any element of `arrays`. If weights=None, 
        then all data in each element of `arrays` are assumed to have a weight equal to one.
    
    Yields
    ------
    var: `~numpy.ndarray`
        Variance. 
    
    See Also
    --------
    numpy.var : variance calculation for dense arrays. Weights are not supported.
    
    References
    ----------
    .. [#] D. H. D. West, Updating the mean and variance estimates: an improved method.
        Communications of the ACM Vol. 22, Issue 9, pp. 532 - 535 (1979)
    """
    yield from ivar(arrays, axis = axis, ddof = ddof, weights = weights, ignore_nan = True)

def istd(arrays, axis = -1, ddof = 0, weights = None, ignore_nan = False):
    """ 
    Streaming standard deviation of arrays. Weights are also supported.
    This is equivalent to calling `numpy.std(axis = 2)` on a stack of images.

    Parameters
    ----------
    arrays : iterable of ndarrays
        Arrays to be averaged. This iterable can also a generator.
    axis : int, optional
        Reduction axis. Default is to combine the arrays in the stream as if 
        they had been stacked along a new axis, then compute the standard deviation along this new axis.
        If None, arrays are flattened. If `axis` is an int larger that
        the number of dimensions in the arrays of the stream, standard deviation is computed
        along the new axis.
    ddof : int, optional
        Means Delta Degrees of Freedom.  The divisor used in calculations
        is ``N - ddof``, where ``N`` represents the number of elements.
        By default `ddof` is one.
    weights : iterable of ndarray, iterable of floats, or None, optional
        Iterable of weights associated with the values in each item of `arrays`. 
        Each value in an element of `arrays` contributes to the variance 
        according to its associated weight. The weights array can either be a float
        or an array of the same shape as any element of `arrays`. If weights=None, 
        then all data in each element of `arrays` are assumed to have a weight equal to one.
    ignore_nan : bool, optional
        If True, NaNs are set to zero weight. Default is propagation of NaNs.
    
    Yields
    ------
    std: `~numpy.ndarray`
        Standard deviation

    See Also
    --------
    numpy.std : standard deviation calculation of dense arrays. Weights are not supported.
    """
    yield from map(np.sqrt, ivar(arrays, axis = axis, ddof = ddof, 
                                 weights = weights, ignore_nan = ignore_nan))

def inanstd(arrays, axis = -1, ddof = 0, weights = None):
    """ 
    Streaming standard deviation of arrays. Weights are also supported.
    NaNs are ignored. Equivalent to `istd(ignore_nan = True)`

    Parameters
    ----------
    arrays : iterable of ndarrays
        Arrays to be averaged. This iterable can also a generator.
    axis : int, optional
        Reduction axis. Default is to combine the arrays in the stream as if 
        they had been stacked along a new axis, then compute the standard deviation along this new axis.
        If None, arrays are flattened. If `axis` is an int larger that
        the number of dimensions in the arrays of the stream, standard deviation is computed
        along the new axis.
    ddof : int, optional
        Means Delta Degrees of Freedom.  The divisor used in calculations
        is ``N - ddof``, where ``N`` represents the number of elements.
        By default `ddof` is one.
    weights : iterable of ndarray, iterable of floats, or None, optional
        Iterable of weights associated with the values in each item of `arrays`. 
        Each value in an element of `arrays` contributes to the variance 
        according to its associated weight. The weights array can either be a float
        or an array of the same shape as any element of `arrays`. If weights=None, 
        then all data in each element of `arrays` are assumed to have a weight equal to one.
    ignore_nan : bool, optional
        If True, NaNs are set to zero weight. Default is propagation of NaNs.
    
    Yields
    ------
    std: `~numpy.ndarray`
        Standard deviation

    See Also
    --------
    numpy.std : standard deviation calculation of dense arrays. Weights are not supported.
    """
    yield from istd(arrays, axis = axis, ddof = ddof, weights = weights, ignore_nan = True)

def isem(arrays, axis = -1, ddof = 1, ignore_nan = False):
    """ 
    Streaming standard error in the mean (SEM) of arrays. This is equivalent to
    calling `scipy.stats.sem(axis = 2)` on a stack of images.

    Parameters
    ----------
    arrays : iterable of ndarrays
        Arrays to be averaged. This iterable can also a generator.
    axis : int, optional
        Reduction axis. Default is to combine the arrays in the stream as if 
        they had been stacked along a new axis, then compute the standard error along this new axis.
        If None, arrays are flattened. If `axis` is an int larger that
        the number of dimensions in the arrays of the stream, standard error is computed
        along the new axis.
    ddof : int, optional
        Means Delta Degrees of Freedom.  The divisor used in calculations
        is ``N - ddof``, where ``N`` represents the number of elements.
        By default `ddof` is one.
    ignore_nan : bool, optional
        If True, NaNs are set to zero weight. Default is propagation of NaNs.
    
    Yields
    ------
    sem: `~numpy.ndarray`
        Standard error in the mean. 
    
    See Also
    --------
    scipy.stats.sem : standard error in the mean of dense arrays.
    """
    # TODO: include weights
    for k, std in enumerate(istd(arrays, axis = axis, ddof = ddof, ignore_nan = ignore_nan), start = 1):
        yield std / sqrt(k) 