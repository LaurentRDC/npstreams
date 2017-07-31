# -*- coding: utf-8 -*-
"""
Statistical functions
---------------------
"""
from itertools import repeat
import numpy as np
from math import sqrt

# TODO: handle NaNs by having array sum_of_weights and not counting NaNs
def iaverage(arrays, weights = None):
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
    
    if weights is None:
        weights = repeat(1.0)
    weights = iter(weights)

    sum_of_weights = np.array(next(weights), copy = True)
    weighted_sum = np.array(next(arrays) * sum_of_weights, copy = True)
    yield weighted_sum/sum_of_weights

    for array, weight in zip(arrays, weights):

        sum_of_weights += weight
        weighted_sum += weight * array
        #print(sum_of_weights)
        yield weighted_sum/sum_of_weights

def imean(arrays):
    """ 
    Streaming mean of arrays. Equivalent to `iaverage(arrays, weights = None)`.

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
    
    Yields
    ------
    avg: `~numpy.ndarray`
        Weighted average. 
    
    See Also
    --------
    iaverage : streaming (weighted) average
    numpy.average : (weighted) average for dense arrays
    """
    yield from iaverage(arrays, weights = repeat(1.0))

def ivar(arrays, ddof = 1, weights = None):
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

    if weights is None:
        weights = repeat(1.0)
    weights = iter(weights)
    sum_of_weights = np.array(next(weights), copy = True)

    first = next(arrays)
    old_mean = new_mean = np.array(first, copy = True)
    old_S = new_S = np.zeros_like(first, dtype = np.float)
    yield np.zeros_like(first)  # No error if no averaging
    
    for array, weight in zip(arrays, weights):

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

def istd(arrays, ddof = 1, weights = None):
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
    
    Yields
    ------
    std: `~numpy.ndarray`
        Standard deviation

    See Also
    --------
    numpy.std : standard deviation calculation of dense arrays. Weights are not supported.
    """
    yield from map(np.sqrt, ivar(arrays, ddof = ddof, weights = weights))

def isem(arrays, ddof = 1):
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
    
    Yields
    ------
    sem: `~numpy.ndarray`
        Standard error in the mean. 
    
    See Also
    --------
    scipy.stats.sem : standard error in the mean of dense arrays.
    """
    # TODO: include weights
    for k, std in enumerate(istd(arrays, ddof = ddof), start = 1):
        yield std / sqrt(k) 