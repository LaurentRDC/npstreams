# -*- coding: utf-8 -*-
"""
Statistical functions
---------------------
"""
from functools import partial
from itertools import count, repeat, starmap
from operator import truediv

import numpy as np

from .array_stream import array_stream
from .array_utils import nan_to_num
from .iter_utils import itercopy, last, peek
from .numerics import isum


@array_stream
def _iaverage(arrays, axis=-1, weights=None, ignore_nan=False):
    """ 
    Primitive version of weighted averaging that yields the running sum and running weights sum,
    but avoids the costly division at every step.
    """
    # Special case: in the easiest case, no need to calculate
    # weights and ignore nans.
    # This case is pretty common
    if (weights is None) and (not ignore_nan) and (axis == -1):
        yield from zip(
            isum(arrays, axis=axis, dtype=np.float, ignore_nan=False), count(1)
        )
        return

    first, arrays = peek(arrays)

    # We make sure that weights is always an array
    # This simplifies the handling of NaNs.
    if weights is None:
        weights = repeat(1)
    weights = map(partial(np.broadcast_to, shape=first.shape), weights)

    # Need to know which array has NaNs, and modify the weights stream accordingly
    if ignore_nan:
        arrays, arrays2 = itercopy(arrays)
        weights = map(
            lambda arr, wgt: np.logical_not(np.isnan(arr)) * wgt, arrays2, weights
        )

    weights1, weights2 = itercopy(weights)

    sum_of_weights = isum(weights1, axis=axis, dtype=np.float)
    weighted_arrays = map(lambda arr, wgt: arr * wgt, arrays, weights2)
    weighted_sum = isum(
        weighted_arrays, axis=axis, ignore_nan=ignore_nan, dtype=np.float
    )

    yield from zip(weighted_sum, sum_of_weights)


@array_stream
def average(arrays, axis=-1, weights=None, ignore_nan=False):
    """ 
    Average (weighted) of a stream of arrays. This function consumes the
    entire stream.

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
        Iterable of weights associated with the values in each item of `arrays`. 
        Each value in an element of `arrays` contributes to the average 
        according to its associated weight. The weights array can either be a float
        or an array of the same shape as any element of `arrays`. If ``weights=None``, 
        then all data in each element of `arrays` are assumed to have a weight equal to one.
    ignore_nan : bool, optional
        If True, NaNs are set to zero weight. Default is propagation of NaNs.
    
    Returns
    -------
    avg: `~numpy.ndarray`, dtype float
        Weighted average. 
    
    See Also
    --------
    iaverage : streaming (weighted) average.
    numpy.average : (weighted) average of dense arrays
    mean : non-weighted average of a stream.
    """
    total_sum, total_weight = last(_iaverage(arrays, axis, weights, ignore_nan))
    return total_sum / total_weight


@array_stream
def iaverage(arrays, axis=-1, weights=None, ignore_nan=False):
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
        Iterable of weights associated with the values in each item of `arrays`. 
        Each value in an element of `arrays` contributes to the average 
        according to its associated weight. The weights array can either be a float
        or an array of the same shape as any element of `arrays`. If weights=None, 
        then all data in each element of `arrays` are assumed to have a weight equal to one.
    ignore_nan : bool, optional
        If True, NaNs are set to zero weight. Default is propagation of NaNs.
    
    Yields
    ------
    avg: `~numpy.ndarray`, dtype float
        Weighted average. 
    
    See Also
    --------
    imean : streaming array mean (non-weighted average).
    """
    # Primitive stream is composed of tuples (running_sum, running_weights)
    primitive = _iaverage(arrays, axis, weights, ignore_nan)
    yield from map(lambda element: truediv(*element), primitive)


@array_stream
def mean(arrays, axis=-1, ignore_nan=False):
    """ 
    Mean of a stream of arrays. This function consumes the
    entire stream.

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
    
    Returns
    -------
    mean: `~numpy.ndarray`, dtype float
        Total mean array.
    """
    total_sum, total_count = last(
        _iaverage(arrays, axis, weights=None, ignore_nan=ignore_nan)
    )
    return total_sum / total_count


@array_stream
def imean(arrays, axis=-1, ignore_nan=False):
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
    mean: `~numpy.ndarray`, dtype float
        Online mean array.
    """
    # Primitive stream is composed of tuples (running_sum, running_count)
    primitive = _iaverage(arrays, axis, weights=None, ignore_nan=ignore_nan)
    yield from map(lambda element: truediv(*element), primitive)


@array_stream
def _ivar(arrays, axis=-1, weights=None, ignore_nan=False):
    """ 
    Primitive version of weighted variance that yields the running average, running average of squares and running weights sum,
    but avoids the costly division and squaring at every step.
    """
    first, arrays = peek(arrays)

    # We make sure that weights is always an array
    # This simplifies the handling of NaNs.
    if weights is None:
        weights = repeat(1)
    weights = map(partial(np.broadcast_to, shape=first.shape), weights)

    # Need to know which array has NaNs, and modify the weights stream accordingly
    if ignore_nan:
        arrays, arrays2 = itercopy(arrays)
        weights = map(
            lambda arr, wgt: np.logical_not(np.isnan(arr)) * wgt, arrays2, weights
        )

    arrays, arrays2 = itercopy(arrays)
    weights, weights2, weights3 = itercopy(weights, 3)

    avgs = iaverage(arrays, axis=axis, weights=weights, ignore_nan=ignore_nan)
    avg_of_squares = iaverage(
        map(np.square, arrays2), axis=axis, weights=weights2, ignore_nan=ignore_nan
    )
    sum_of_weights = isum(weights3, axis=axis, ignore_nan=ignore_nan)

    yield from zip(avgs, avg_of_squares, sum_of_weights)


@array_stream
def average_and_var(arrays, axis=-1, ddof=0, weights=None, ignore_nan=False):
    """
    Calculate the simultaneous average and variance of a stream of arrays. This is done in
    single iteration for maximum performance.

    .. versionadded:: 1.6.1

    Parameters
    ----------
    arrays : iterable of ndarrays
        Arrays to be combined. This iterable can also a generator.
    axis : int, optional
        Reduction axis. Default is to combine the arrays in the stream as if 
        they had been stacked along a new axis, then compute the variance along this new axis.
        If None, arrays are flattened. If `axis` is an int larger that
        the number of dimensions in the arrays of the stream, variance is computed
        along the new axis.
    ddof : int, optional
        Means Delta Degrees of Freedom. The divisor used in calculations
        is ``N - ddof``, where ``N`` represents the number of elements.
    weights : iterable of ndarray, iterable of floats, or None, optional
        Iterable of weights associated with the values in each item of `arrays`. 
        Each value in an element of `arrays` contributes to the variance 
        according to its associated weight. The weights array can either be a float
        or an array of the same shape as any element of `arrays`. If weights=None, 
        then all data in each element of `arrays` are assumed to have a weight equal to one.
    ignore_nan : bool, optional
        If True, NaNs are set to zero weight. Default is propagation of NaNs.
    
    Returns
    -------
    average : `~numpy.ndarray`
        Average, possibly weighted.
    var: `~numpy.ndarray`
        Variance, possibly weighted.
    
    Notes
    -----
    Since the calculation of the variance requires knowledge of the average, this function is a
    very thin wrapper around `var`.

    References
    ----------
    .. [#] D. H. D. West, Updating the mean and variance estimates: an improved method.
        Communications of the ACM Vol. 22, Issue 9, pp. 532 - 535 (1979)
    """
    # Since the variance calculation requires knowing the average,
    # `average_and_var` runs in the exact same time as `var`
    avg, sq_avg, swgt = last(
        _ivar(arrays=arrays, axis=axis, weights=weights, ignore_nan=ignore_nan)
    )
    variance = (sq_avg - avg ** 2) * (swgt / (swgt - ddof))
    return avg, variance


@array_stream
def var(arrays, axis=-1, ddof=0, weights=None, ignore_nan=False):
    """ 
    Total variance of a stream of arrays. Weights are also supported. This function
    consumes the input stream.
    
    Parameters
    ----------
    arrays : iterable of ndarrays
        Arrays to be combined. This iterable can also a generator.
    axis : int, optional
        Reduction axis. Default is to combine the arrays in the stream as if 
        they had been stacked along a new axis, then compute the variance along this new axis.
        If None, arrays are flattened. If `axis` is an int larger that
        the number of dimensions in the arrays of the stream, variance is computed
        along the new axis.
    ddof : int, optional
        Means Delta Degrees of Freedom.  The divisor used in calculations
        is ``N - ddof``, where ``N`` represents the number of elements.
    weights : iterable of ndarray, iterable of floats, or None, optional
        Iterable of weights associated with the values in each item of `arrays`. 
        Each value in an element of `arrays` contributes to the variance 
        according to its associated weight. The weights array can either be a float
        or an array of the same shape as any element of `arrays`. If weights=None, 
        then all data in each element of `arrays` are assumed to have a weight equal to one.
    ignore_nan : bool, optional
        If True, NaNs are set to zero weight. Default is propagation of NaNs.
    
    Returns
    -------
    var: `~numpy.ndarray`
        Variance. 
    
    See Also
    --------
    ivar : streaming variance
    numpy.var : variance calculation for dense arrays. Weights are not supported.
    
    References
    ----------
    .. [#] D. H. D. West, Updating the mean and variance estimates: an improved method.
        Communications of the ACM Vol. 22, Issue 9, pp. 532 - 535 (1979)
    """
    _, variance = average_and_var(
        arrays=arrays, axis=axis, ddof=ddof, weights=weights, ignore_nan=ignore_nan
    )
    return variance


@array_stream
def ivar(arrays, axis=-1, ddof=0, weights=None, ignore_nan=False):
    """ 
    Streaming variance of arrays. Weights are also supported.
    
    Parameters
    ----------
    arrays : iterable of ndarrays
        Arrays to be combined. This iterable can also a generator.
    axis : int, optional
        Reduction axis. Default is to combine the arrays in the stream as if 
        they had been stacked along a new axis, then compute the variance along this new axis.
        If None, arrays are flattened. If `axis` is an int larger that
        the number of dimensions in the arrays of the stream, variance is computed
        along the new axis.
    ddof : int, optional
        Means Delta Degrees of Freedom.  The divisor used in calculations
        is ``N - ddof``, where ``N`` represents the number of elements.
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
    primitive = _ivar(arrays=arrays, axis=axis, weights=weights, ignore_nan=ignore_nan)
    for avg, sq_avg, swgt in primitive:
        yield (sq_avg - avg ** 2) * (swgt / (swgt - ddof))


@array_stream
def std(arrays, axis=-1, ddof=0, weights=None, ignore_nan=False):
    """ 
    Total standard deviation of arrays. Weights are also supported. This function
    consumes the input stream.

    Parameters
    ----------
    arrays : iterable of ndarrays
        Arrays to be combined. This iterable can also a generator.
    axis : int, optional
        Reduction axis. Default is to combine the arrays in the stream as if 
        they had been stacked along a new axis, then compute the standard deviation along this new axis.
        If None, arrays are flattened. If `axis` is an int larger that
        the number of dimensions in the arrays of the stream, standard deviation is computed
        along the new axis.
    ddof : int, optional
        Means Delta Degrees of Freedom.  The divisor used in calculations
        is ``N - ddof``, where ``N`` represents the number of elements.
    weights : iterable of ndarray, iterable of floats, or None, optional
        Iterable of weights associated with the values in each item of `arrays`. 
        Each value in an element of `arrays` contributes to the standard deviation 
        according to its associated weight. The weights array can either be a float
        or an array of the same shape as any element of `arrays`. If weights=None, 
        then all data in each element of `arrays` are assumed to have a weight equal to one.
    ignore_nan : bool, optional
        If True, NaNs are set to zero weight. Default is propagation of NaNs.
    
    Returns
    -------
    std: `~numpy.ndarray`
        Standard deviation

    See Also
    --------
    istd : streaming standard deviation.
    numpy.std : standard deviation calculation of dense arrays. Weights are not supported.
    """
    return np.sqrt(
        var(arrays=arrays, axis=axis, ddof=ddof, weights=weights, ignore_nan=ignore_nan)
    )


@array_stream
def istd(arrays, axis=-1, ddof=0, weights=None, ignore_nan=False):
    """ 
    Streaming standard deviation of arrays. Weights are also supported.
    This is equivalent to calling `numpy.std(axis = 2)` on a stack of images.

    Parameters
    ----------
    arrays : iterable of ndarrays
        Arrays to be combined. This iterable can also a generator.
    axis : int, optional
        Reduction axis. Default is to combine the arrays in the stream as if 
        they had been stacked along a new axis, then compute the standard deviation along this new axis.
        If None, arrays are flattened. If `axis` is an int larger that
        the number of dimensions in the arrays of the stream, standard deviation is computed
        along the new axis.
    ddof : int, optional
        Means Delta Degrees of Freedom.  The divisor used in calculations
        is ``N - ddof``, where ``N`` represents the number of elements.
    weights : iterable of ndarray, iterable of floats, or None, optional
        Iterable of weights associated with the values in each item of `arrays`. 
        Each value in an element of `arrays` contributes to the standard deviation 
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
    std : total standard deviation.
    numpy.std : standard deviation calculation of dense arrays. Weights are not supported.
    """
    yield from map(
        np.sqrt,
        ivar(
            arrays=arrays, axis=axis, ddof=ddof, weights=weights, ignore_nan=ignore_nan
        ),
    )


@array_stream
def sem(arrays, axis=-1, ddof=0, weights=None, ignore_nan=False):
    """ 
    Standard error in the mean (SEM) of a stream of arrays. This function consumes
    the entire stream.

    Parameters
    ----------
    arrays : iterable of ndarrays
        Arrays to be combined. This iterable can also a generator.
    axis : int, optional
        Reduction axis. Default is to combine the arrays in the stream as if 
        they had been stacked along a new axis, then compute the standard error along this new axis.
        If None, arrays are flattened. If `axis` is an int larger that
        the number of dimensions in the arrays of the stream, standard error is computed
        along the new axis.
    ddof : int, optional
        Means Delta Degrees of Freedom.  The divisor used in calculations
        is ``N - ddof``, where ``N`` represents the number of elements.
    weights : iterable of ndarray, iterable of floats, or None, optional
        Iterable of weights associated with the values in each item of `arrays`. 
        Each value in an element of `arrays` contributes to the standard error 
        according to its associated weight. The weights array can either be a float
        or an array of the same shape as any element of `arrays`. If weights=None, 
        then all data in each element of `arrays` are assumed to have a weight equal to one.
    ignore_nan : bool, optional
        If True, NaNs are set to zero weight. Default is propagation of NaNs.
    
    Returns
    -------
    sem: `~numpy.ndarray`, dtype float
        Standard error in the mean. 
    
    See Also
    --------
    scipy.stats.sem : standard error in the mean of dense arrays.
    """
    avg, sq_avg, swgt = last(
        _ivar(arrays=arrays, axis=axis, weights=weights, ignore_nan=ignore_nan)
    )
    return np.sqrt((sq_avg - avg ** 2) * (1 / (swgt - ddof)))


@array_stream
def isem(arrays, axis=-1, ddof=1, weights=None, ignore_nan=False):
    """ 
    Streaming standard error in the mean (SEM) of arrays. This is equivalent to
    calling `scipy.stats.sem(axis = 2)` on a stack of images.

    Parameters
    ----------
    arrays : iterable of ndarrays
        Arrays to be combined. This iterable can also a generator.
    axis : int, optional
        Reduction axis. Default is to combine the arrays in the stream as if 
        they had been stacked along a new axis, then compute the standard error along this new axis.
        If None, arrays are flattened. If `axis` is an int larger that
        the number of dimensions in the arrays of the stream, standard error is computed
        along the new axis.
    ddof : int, optional
        Means Delta Degrees of Freedom.  The divisor used in calculations
        is ``N - ddof``, where ``N`` represents the number of elements.
    weights : iterable of ndarray, iterable of floats, or None, optional
        Iterable of weights associated with the values in each item of `arrays`. 
        Each value in an element of `arrays` contributes to the standard error 
        according to its associated weight. The weights array can either be a float
        or an array of the same shape as any element of `arrays`. If weights=None, 
        then all data in each element of `arrays` are assumed to have a weight equal to one.
    ignore_nan : bool, optional
        If True, NaNs are set to zero weight. Default is propagation of NaNs.
    
    Yields
    ------
    sem: `~numpy.ndarray`, dtype float
        Standard error in the mean. 
    
    See Also
    --------
    scipy.stats.sem : standard error in the mean of dense arrays.
    """
    primitive = _ivar(arrays=arrays, axis=axis, weights=weights, ignore_nan=ignore_nan)
    for avg, sq_avg, swgt in primitive:
        yield np.sqrt((sq_avg - avg ** 2) * (1 / (swgt - ddof)))


@array_stream
def ihistogram(arrays, bins, range=None, weights=None):
    """
    Streaming histogram calculation.

    Parameters
    ----------
    arrays : iterable of ndarrays
        Arrays to be combined. This iterable can also a generator. Arrays in this stream
        can be of any shape; the histogram is computed over the flattened array.
    bins : iterable
        Bin edges, including the rightmost edge, allowing for non-uniform bin widths.
        To determine the appropriate bins automatically, see ``numpy.histogram_bin_edges``.
    weights : iterable of ndarray, iterable of floats, or None, optional
        Iterable of weights associated with the values in each item of `arrays`. 
        Each value in a only contributes its associated weight towards the 
        bin count (instead of 1). The weights array can either be a float
        or an array of the same shape as any element of `arrays`. If ``weights=None``, 
        then all data in each element of `arrays` are assumed to have a weight equal to one.

        .. versionadded:: 1.6.1

    Yields
    ------
    hist : `~numpy.ndarray`
        Streamed histogram.
    
    See Also
    --------
    numpy.histogram : 1D histogram of dense arrays.
    numpy.histogram_bin_edges : automatic selection of bins
    """
    bins = np.asarray(bins)
    first, arrays = peek(arrays)

    if weights is None:
        weights = repeat(None)
    else:
        weights = map(partial(np.broadcast_to, shape=first.shape), weights)

    # np.histogram also returns the bin edges, which we ignore
    hist_func = lambda arr, wgt: np.histogram(arr, bins=bins, weights=wgt)[0]
    yield from isum(starmap(hist_func, zip(arrays, weights)))
