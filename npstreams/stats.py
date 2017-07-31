# -*- coding: utf-8 -*-
"""
Statistical functions
---------------------
"""
from functools import partial
from itertools import repeat, tee
import numpy as np
from math import sqrt
from . import _nan_to_num

def _atleast_array(arg, arr):
    """ Make sure that if inputs are float or int, they are array of shape `shape` """
    if isinstance(arg, (float, int)):
        arg = np.full(shape = arr.shape, fill_value = arg, dtype = arr.dtype)
    else:
        arg = np.asarray(arg)
    return arg

# TODO: handle NaNs by having array sum_of_weights and not counting NaNs
def iaverage(arrays, weights = None, ignore_nan = False):
    """ 
    Streaming (weighted) average of arrays.

    Parameters
    ----------
    arrays : iterable of ndarrays
        Arrays to be averaged. This iterable can also a generator.
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
    arrays = iter(arrays)
    first = next(arrays)
    
    # We make sure that weights is always an array
    # This simplifies the handling of NaNs.
    if weights is None:
        weights = repeat(1)
    weights = map(partial(_atleast_array, arr = first), iter(weights))

    sum_of_weights = np.array(next(weights), copy = True)
    if ignore_nan:
        sum_of_weights[np.isnan(first)] = 0
        first = np.nan_to_num(first)
    weighted_sum = np.array(first * sum_of_weights, copy = True)
    yield weighted_sum/sum_of_weights

    for array, weight in zip(arrays, weights):
        valid = np.s_[:]
        if ignore_nan:
            valid = np.logical_not(np.isnan(array))

        sum_of_weights[valid] += weight[valid]
        weighted_sum[valid] += weight[valid] * array[valid]
        yield weighted_sum/sum_of_weights

def imean(arrays, ignore_nan = False):
    """ 
    Streaming mean of arrays. Equivalent to `iaverage(arrays, weights = None)`.

    Parameters
    ----------
    arrays : iterable of ndarrays
        Arrays to be averaged. This iterable can also a generator.
    ignore_nan : bool, optional
        If True, NaNs are set to zero weight. Default is propagation of NaNs.
    
    Yields
    ------
    mean: `~numpy.ndarray`
        Online mean array.
    """
    yield from iaverage(arrays, weights = None, ignore_nan = ignore_nan)

def inanmean(arrays):
    """ 
    Streaming mean of arrays, ignoring NaNs. Equivalent to `imean(ignore_nan = True)`.

    Parameters
    ----------
    arrays : iterable of ndarrays
        Arrays to be averaged. This iterable can also a generator.
    
    Yields
    ------
    mean: `~numpy.ndarray`
        Online mean array.
    """
    yield from imean(arrays, ignore_nan = True)

def ivar(arrays, ddof = 0, weights = None, ignore_nan = False):
    """ 
    Streaming variance of arrays. Weights are also supported.
    
    Parameters
    ----------
    arrays : iterable of ndarrays
        Arrays to be averaged. This iterable can also a generator.
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
    
    sum_of_weights = np.array(next(weights), copy = True)
    if ignore_nan:
        sum_of_weights[np.isnan(first)] = 0
        first = np.nan_to_num(first)

    old_mean = np.array(first, copy = True)
    new_mean = np.array(first, copy = True)
    old_S = np.zeros_like(first, dtype = np.float)
    new_S = np.zeros_like(first, dtype = np.float)
    yield np.zeros_like(first)
    
    for array, weight in zip(arrays, weights):

        if ignore_nan:
            weight[np.isnan(array)] = 0
            array = np.nan_to_num(array)

        sum_of_weights += weight

        _sub = weight * (array - old_mean)
        new_mean[:] = old_mean + _sub/sum_of_weights
        new_S[:] = old_S + _sub*(array - new_mean)

        # In case there hasn't been enough measurements yet,
        # yield zeros.
        if np.any(sum_of_weights - ddof <= 0):
            yield np.zeros_like(first)
        else:
            yield new_S/(sum_of_weights - ddof) # variance = S / k-1, sem = std / sqrt(k)    

        old_mean[:] = new_mean
        old_S[:] = new_S

def inanvar(arrays, ddof = 1, weights = None):
    """ 
    Streaming variance of arrays. Weights are also supported. NaNs are ignored.
    Equivalent to `ivarignore_nan = True)`.
    
    Parameters
    ----------
    arrays : iterable of ndarrays
        Arrays to be averaged. This iterable can also a generator.
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
    yield from ivar(arrays, ddof = ddof, weights = weights, ignore_nan = True)

def istd(arrays, ddof = 1, weights = None, ignore_nan = False):
    """ 
    Streaming standard deviation of images. Weights are also supported.
    This is equivalent to calling `numpy.std(axis = 2)` on a stack of images.

    Parameters
    ----------
    arrays : iterable of ndarrays
        Arrays to be averaged. This iterable can also a generator.
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
    yield from map(np.sqrt, ivar(arrays, ddof = ddof, weights = weights, ignore_nan = ignore_nan))

def inanstd(arrays, ddof = 1, weights = None):
    """ 
    Streaming standard deviation of images. Weights are also supported.
    NaNs are ignored. Equivalent to `istd(ignore_nan = True)`

    Parameters
    ----------
    arrays : iterable of ndarrays
        Arrays to be averaged. This iterable can also a generator.
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
    yield from istd(arrays, ddof = ddof, weights = weights, ignore_nan = True)

def isem(arrays, ddof = 1, ignore_nan = False):
    """ 
    Streaming standard error in the mean (SEM) of images. This is equivalent to
    calling `scipy.stats.sem(axis = 2)` on a stack of images.

    Parameters
    ----------
    arrays : iterable of ndarrays
        Arrays to be averaged. This iterable can also a generator.
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
    for k, std in enumerate(istd(arrays, ddof = ddof, ignore_nan = ignore_nan), start = 1):
        yield std / sqrt(k) 