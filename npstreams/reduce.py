# -*- coding: utf-8 -*-
"""
General stream reduction
------------------------
"""
from functools import lru_cache, partial
from itertools import islice, repeat
from multiprocessing import Pool

import numpy as np

from .array_stream import array_stream
from .array_utils import nan_to_num
from .iter_utils import chunked, last, peek, primed
from .parallel import preduce


@lru_cache(maxsize=128)
def _check_binary_ufunc(ufunc):
    """ 
    Check that ufunc is suitable for ``ireduce_ufunc``. 
    
    Specifically, a binary ``numpy.ufunc`` function is required. Functions 
    that returns a boolean are also not suitable because they cannot be accumulated.

    This function does not return anything. 

    Parameters
    ----------
    ufunc : callable
        Function to check.

    Raises
    ------
    TypeError : if ``ufunc`` is not a ``numpy.ufunc``
    ValueError: if ``ufunc`` is not binary or the return type is boolean.
    """
    if not isinstance(ufunc, np.ufunc):
        raise TypeError("{} is not a NumPy Ufunc".format(ufunc.__name__))
    if ufunc.nin != 2:
        raise ValueError(
            "Only binary ufuncs are supported, and {} is \
                          not one of them".format(
                ufunc.__name__
            )
        )

    # Ufuncs that always return bool are problematic because they can be reduced
    # but not be accumulated.
    # Recall: numpy.dtype('?') == np.bool
    if all(type_signature[-1] == "?" for type_signature in ufunc.types):
        raise ValueError(
            "Only binary ufuncs that preserve type are supported, \
                          and {} is not one of them".format(
                ufunc.__name__
            )
        )


@primed
@array_stream
def ireduce_ufunc(arrays, ufunc, axis=-1, dtype=None, ignore_nan=False, **kwargs):
    """
    Streaming reduction generator function from a binary NumPy ufunc. Generator
    version of `reduce_ufunc`.

    ``ufunc`` must be a NumPy binary Ufunc (i.e. it takes two arguments). Moreover,
    for performance reasons, ufunc must have the same return types as input types.
    This precludes the use of ``numpy.greater``, for example.

    Note that performance is much better for the default ``axis = -1``. In such a case,
    reduction operations can occur in-place. This also allows to operate in constant-memory.
    
    Parameters
    ----------
    arrays : iterable
        Arrays to be reduced.
    ufunc : numpy.ufunc
        Binary universal function.
    axis : int or None, optional
        Reduction axis. Default is to reduce the arrays in the stream as if 
        they had been stacked along a new axis, then reduce along this new axis.
        If None, arrays are flattened before reduction. If `axis` is an int larger that
        the number of dimensions in the arrays of the stream, arrays are reduced
        along the new axis. Note that not all of NumPy Ufuncs support 
        ``axis = None``, e.g. ``numpy.subtract``.
    dtype : numpy.dtype or None, optional
        Overrides the dtype of the calculation and output arrays.
    ignore_nan : bool, optional
        If True and ufunc has an identity value (e.g. ``numpy.add.identity`` is 0), then NaNs
        are replaced with this identity. An error is raised if ``ufunc`` has no identity 
        (e.g. ``numpy.maximum.identity`` is ``None``).
    kwargs
        Keyword arguments are passed to ``ufunc``. Note that some valid ufunc keyword arguments
        (e.g. ``keepdims``) are not valid for all streaming functions. Also, contrary to NumPy 
        v. 1.10+, ``casting = 'unsafe`` is the default in npstreams.
    
    Yields 
    ------
    reduced : ndarray or scalar

    Raises
    ------
    TypeError : if ``ufunc`` is not NumPy ufunc.
    ValueError : if ``ignore_nan`` is True but ``ufunc`` has no identity
    ValueError : if ``ufunc`` is not a binary ufunc
    ValueError : if ``ufunc`` does not have the same input type as output type
    """
    kwargs.update({"dtype": dtype, "axis": axis})

    _check_binary_ufunc(ufunc)

    if ignore_nan:
        if ufunc.identity is None:
            raise ValueError(
                "Cannot ignore NaNs because {} has no identity value".format(
                    ufunc.__name__
                )
            )
        arrays = map(partial(nan_to_num, fill_value=ufunc.identity, copy=False), arrays)

    # Since ireduce_ufunc is primed, we need to wait here
    # Priming is a way to start error checking before actually running
    # any computations.
    yield

    if kwargs["axis"] == -1:
        yield from _ireduce_ufunc_new_axis(arrays, ufunc, **kwargs)
        return

    if kwargs["axis"] is None:
        yield from _ireduce_ufunc_all_axes(arrays, ufunc, **kwargs)
        return

    first, arrays = peek(arrays)

    if kwargs["axis"] >= first.ndim:
        kwargs["axis"] = -1
        yield from ireduce_ufunc(arrays, ufunc, **kwargs)
        return

    yield from _ireduce_ufunc_existing_axis(arrays, ufunc, **kwargs)


def reduce_ufunc(arrays, ufunc, axis=-1, dtype=None, ignore_nan=False, **kwargs):
    """
    Reduce a stream using a binary NumPy ufunc. Function version of ``ireduce_ufunc``.

    ``ufunc`` must be a NumPy binary Ufunc (i.e. it takes two arguments). Moreover,
    for performance reasons, ufunc must have the same return types as input types.
    This precludes the use of ``numpy.greater``, for example.

    Note that performance is much better for the default ``axis = -1``. In such a case,
    reduction operations can occur in-place. This also allows to operate in constant-memory.
    
    Parameters
    ----------
    arrays : iterable
        Arrays to be reduced.
    ufunc : numpy.ufunc
        Binary universal function.
    axis : int or None, optional
        Reduction axis. Default is to reduce the arrays in the stream as if 
        they had been stacked along a new axis, then reduce along this new axis.
        If None, arrays are flattened before reduction. If `axis` is an int larger that
        the number of dimensions in the arrays of the stream, arrays are reduced
        along the new axis. Note that not all of NumPy Ufuncs support 
        ``axis = None``, e.g. ``numpy.subtract``.
    dtype : numpy.dtype or None, optional
        Overrides the dtype of the calculation and output arrays.
    ignore_nan : bool, optional
        If True and ufunc has an identity value (e.g. ``numpy.add.identity`` is 0), then NaNs
        are replaced with this identity. An error is raised if ``ufunc`` has no identity (e.g. ``numpy.maximum.identity`` is ``None``).
    kwargs
        Keyword arguments are passed to ``ufunc``. Note that some valid ufunc keyword arguments
        (e.g. ``keepdims``) are not valid for all streaming functions. Note that
        contrary to NumPy v. 1.10+, ``casting = 'unsafe`` is the default in npstreams.
    
    Returns 
    -------
    reduced : ndarray or scalar

    Raises
    ------
    TypeError : if ``ufunc`` is not NumPy ufunc.
    ValueError : if ``ignore_nan`` is True but ``ufunc`` has no identity
    ValueError: if ``ufunc`` is not a binary ufunc
    ValueError: if ``ufunc`` does not have the same input type as output type
    """
    return last(
        ireduce_ufunc(
            arrays, ufunc, axis=axis, dtype=dtype, ignore_nan=ignore_nan, **kwargs
        )
    )


@array_stream
def preduce_ufunc(
    arrays,
    ufunc,
    axis=-1,
    dtype=None,
    ignore_nan=False,
    processes=1,
    ntotal=None,
    **kwargs
):
    """
    Parallel reduction of array streams.

    ``ufunc`` must be a NumPy binary Ufunc (i.e. it takes two arguments). Moreover,
    for performance reasons, ufunc must have the same return types as input types.
    This precludes the use of ``numpy.greater``, for example.

    Parameters
    ----------
    arrays : iterable
        Arrays to be reduced.
    ufunc : numpy.ufunc
        Binary universal function.
    axis : int or None, optional
        Reduction axis. Default is to reduce the arrays in the stream as if 
        they had been stacked along a new axis, then reduce along this new axis.
        If None, arrays are flattened before reduction. If `axis` is an int larger that
        the number of dimensions in the arrays of the stream, arrays are reduced
        along the new axis. Note that not all of NumPy Ufuncs support 
        ``axis = None``, e.g. ``numpy.subtract``.
    dtype : numpy.dtype or None, optional
        Overrides the dtype of the calculation and output arrays.
    ignore_nan : bool, optional
        If True and ufunc has an identity value (e.g. ``numpy.add.identity`` is 0), then NaNs
        are replaced with this identity. An error is raised if ``ufunc`` has no identity (e.g. ``numpy.maximum.identity`` is ``None``).
    processes : int or None, optional
        Number of processes to use. If `None`, maximal number of processes
        is used. Default is 1.
    kwargs
        Keyword arguments are passed to ``ufunc``. Note that some valid ufunc keyword arguments
        (e.g. ``keepdims``) are not valid for all streaming functions. Also, contrary to NumPy 
        v. 1.10+, ``casting = 'unsafe`` is the default in npstreams.
    """
    if processes == 1:
        return reduce_ufunc(arrays, ufunc, axis, dtype, ignore_nan, **kwargs)

    kwargs.update(
        {"ufunc": ufunc, "ignore_nan": ignore_nan, "dtype": dtype, "axis": axis}
    )
    reduce = partial(reduce_ufunc, **kwargs)
    # return preduce(reduce, arrays, processes = processes, ntotal = ntotal)

    with Pool(processes) as pool:
        chunksize = 1
        if ntotal is not None:
            chunksize = max(1, int(ntotal / pool._processes))
        res = pool.imap(reduce, chunked(arrays, chunksize))
        return reduce(res)


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
        Keyword arguments are passed to ``ufunc``.
    
    Yields 
    ------
    reduced : ndarray
    """
    arrays = iter(arrays)
    first = next(arrays)

    kwargs.pop("axis")

    dtype = kwargs.get("dtype", None)
    if dtype is None:
        dtype = first.dtype
    else:
        kwargs["casting"] = "unsafe"

    # If the out parameter was already given
    # we create the accumulator from it
    # Otherwise, it is a copy of the first array
    accumulator = kwargs.pop("out", None)
    if accumulator is not None:
        accumulator[:] = first
    else:
        accumulator = np.array(first, copy=True).astype(dtype)
    yield accumulator

    for array in arrays:
        ufunc(accumulator, array, out=accumulator, **kwargs)
        yield accumulator


def _ireduce_ufunc_existing_axis(arrays, ufunc, **kwargs):
    """
    Reduction operation for arrays, in the direction of an existing axis.
    
    Parameters
    ----------
    arrays : iterable
        Arrays to be reduced.
    ufunc : numpy.ufunc
        Binary universal function. Must have a signature of the form ufunc(x1, x2, ...)
    kwargs
        Keyword arguments are passed to ``ufunc``. The ``out`` parameter is ignored.

    Yields 
    ------
    reduced : ndarray
    """
    arrays = iter(arrays)
    first = next(arrays)

    if kwargs["axis"] not in range(first.ndim):
        raise ValueError(
            "Axis {} not supported on arrays of shape {}.".format(
                kwargs["axis"], first.shape
            )
        )

    # Remove parameters that will not be used.
    kwargs.pop("out", None)

    dtype = kwargs.get("dtype")
    if dtype is None:
        dtype = first.dtype

    axis_reduce = partial(ufunc.reduce, **kwargs)

    accumulator = np.atleast_1d(axis_reduce(first))
    yield accumulator

    # On the first pass of the following loop, accumulator is missing a dimensions
    # therefore, the stacking function cannot be 'concatenate'
    second = next(arrays)
    accumulator = np.stack([accumulator, np.atleast_1d(axis_reduce(second))], axis=-1)
    yield accumulator

    # On the second pass, the new dimensions exists, and thus we switch to
    # using concatenate.
    for array in arrays:
        reduced = np.expand_dims(
            np.atleast_1d(axis_reduce(array)), axis=accumulator.ndim - 1
        )
        accumulator = np.concatenate([accumulator, reduced], axis=accumulator.ndim - 1)
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
        Keyword arguments are passed to ``ufunc``. The ``out`` parameter is ignored.

    Yields 
    ------
    reduced : scalar
    """
    arrays = iter(arrays)
    first = next(arrays)

    kwargs.pop("out", None)

    kwargs["axis"] = None
    axis_reduce = partial(ufunc.reduce, **kwargs)

    accumulator = axis_reduce(first)
    yield accumulator

    for array in arrays:
        accumulator = axis_reduce([accumulator, axis_reduce(array)])
        yield accumulator
