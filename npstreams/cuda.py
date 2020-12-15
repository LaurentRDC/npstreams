# -*- coding: utf-8 -*-
"""
CUDA-accelerated streaming operations
-------------------------------------
"""
from functools import partial
from itertools import repeat
from operator import iadd, imul
from subprocess import run, PIPE

import numpy as np

from . import array_stream, itercopy, nan_to_num, peek

# Determine if
#   1. pycuda is installed;
#   2. pycuda can compile with nvcc
#   3. a GPU is available

try:
    import pycuda.gpuarray as gpuarray
    import pycuda.autoinit
except ImportError:
    raise ImportError("PyCUDA is not installed. CUDA capabilities are not available.")
else:
    import pycuda.driver as driver
    from pycuda.compiler import SourceModule

# Check if nvcc compiler is installed at all
nvcc_installed = run(["nvcc", "-h"], stdout=PIPE).returncode == 0
if not nvcc_installed:
    raise ImportError("CUDA compiler `nvcc` not installed.")

# Check that nvcc is at least set up properly
# For example, if nvcc is installed but C++ compiler is not in path
try:
    SourceModule("")
except driver.CompileError:
    raise ImportError("CUDA compiler `nvcc` is not properly set up.")

if driver.Device.count() == 0:
    raise ImportError("No GPU is available.")


@array_stream
def cuda_inplace_reduce(arrays, operator, dtype=None, ignore_nan=False, identity=0):
    """
    Inplace reduce on GPU arrays.

    Parameters
    ----------
    arrays : iterable
        Arrays to be reduced.
    operator : callable
        Callable of two arguments. This operator should operate in-place, storing the results into
        the buffer of the first argument, e.g. operator.iadd
    dtype : numpy.dtype, optional
        Arrays of the stream are cast to this dtype before reduction.
    ignore_nan : bool, optional
        If True, NaNs are replaced with ``identity``. Default is propagation of NaNs.
    identity : float, optional
        If ``ignore_nan = True``, NaNs are replaced with this value.

    Returns
    -------
    out : ndarray
    """
    # No need to cast all arrays if ``dtype`` is the same
    # type as the stream
    first, arrays = peek(arrays)
    if (dtype is not None) and (first.dtype != dtype):
        arrays = map(lambda arr: arr.astype(dtype), arrays)

    if ignore_nan:
        arrays = map(partial(nan_to_num, fill_value=identity), arrays)

    acc_gpu = gpuarray.to_gpu(next(arrays))  # Accumulator
    arr_gpu = gpuarray.empty_like(acc_gpu)  # GPU memory location for each array
    for arr in arrays:
        arr_gpu.set(arr)
        operator(acc_gpu, arr_gpu)

    return acc_gpu.get()


def csum(arrays, dtype=None, ignore_nan=False):
    """
    CUDA-enabled sum of stream of arrays. Arrays are summed along
    the streaming axis for performance reasons.

    Parameters
    ----------
    arrays : iterable
        Arrays to be summed.
    ignore_nan : bool, optional
        If True, NaNs are ignored. Default is propagation of NaNs.

    Returns
    -------
    cuda_sum : ndarray

    See Also
    --------
    isum : streaming sum of array elements, possibly along different axes
    """
    return cuda_inplace_reduce(
        arrays, operator=iadd, dtype=dtype, ignore_nan=ignore_nan, identity=0
    )


def cprod(arrays, dtype=None, ignore_nan=False):
    """
    CUDA-enabled product of a stream of arrays. Arrays are multiplied
    along the streaming axis for performance reasons.

    Parameters
    ----------
    arrays : iterable
        Arrays to be multiplied.
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
    return cuda_inplace_reduce(
        arrays, operator=imul, dtype=dtype, ignore_nan=ignore_nan, identity=1
    )


@array_stream
def cmean(arrays, ignore_nan=False):
    """
    CUDA-enabled mean of stream of arrays (i.e. unweighted average). Arrays are averaged
    along the streaming axis for performance reasons.

    Parameters
    ----------
    arrays : iterable of ndarrays
        Arrays to be averaged. This iterable can also a generator.
    ignore_nan : bool, optional
        If True, NaNs are set to zero weight. Default is propagation of NaNs.

    Returns
    -------
    cuda_mean : ndarray

    See also
    --------
    caverage : CUDA-enabled weighted average
    imean : streaming mean of arrays, possibly along different axes
    """
    first, arrays = peek(arrays)

    # Need to know which array has NaNs, and modify the weights stream accordingly
    if ignore_nan:
        arrays, arrays2 = itercopy(arrays)
        weights = map(
            lambda arr, wgt: np.logical_not(np.isnan(arr)) * wgt, arrays2, weights
        )
        arrays = map(np.nan_to_num, arrays)
        return caverage(arrays, weights, ignore_nan=False)

    accumulator = gpuarray.to_gpu(next(arrays))
    array_gpu = gpuarray.empty_like(accumulator)
    num_arrays = 1
    for arr in arrays:
        num_arrays += 1
        array_gpu.set(arr)
        accumulator += array_gpu

    return accumulator.get() / num_arrays


@array_stream
def caverage(arrays, weights=None, ignore_nan=False):
    """
    CUDA-enabled average of stream of arrays, possibly weighted. Arrays are averaged
    along the streaming axis for performance reasons.

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

    Returns
    -------
    cuda_avg : ndarray

    See also
    --------
    iaverage : streaming weighted average, possibly along different axes
    """
    if weights is None:
        return cmean(arrays, ignore_nan)

    first, arrays = peek(arrays)

    # We make sure that weights is always an array
    # This simplifies the handling of NaNs.
    if weights is None:
        weights = repeat(1)
    weights = map(partial(np.broadcast_to, shape=first.shape), weights)
    weights = map(
        lambda arr: arr.astype(first.dtype), weights
    )  # Won't work without this

    # Need to know which array has NaNs, and modify the weights stream accordingly
    if ignore_nan:
        arrays, arrays2 = itercopy(arrays)
        weights = map(
            lambda arr, wgt: np.logical_not(np.isnan(arr)) * wgt, arrays2, weights
        )
        arrays = map(np.nan_to_num, arrays)

    first = next(arrays)
    fst_wgt = next(weights)

    arr_gpu = gpuarray.to_gpu(first * fst_wgt)
    wgt_gpu = gpuarray.to_gpu(fst_wgt)
    for arr, wgt in zip(arrays, weights):
        arr_gpu += gpuarray.to_gpu(arr) * gpuarray.to_gpu(wgt)
        wgt_gpu += gpuarray.to_gpu(wgt)

    arr_gpu /= wgt_gpu
    return arr_gpu.get()
