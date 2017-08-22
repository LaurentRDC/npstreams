# -*- coding: utf-8 -*-
"""
CUDA-accelerated streaming operations
-------------------------------------
"""
from functools import partial
from itertools import repeat
import numpy as np
from warnings import warn

from . import array_stream, peek

# Determine if 
#   1. pycuda is installed;
#   2. pycuda can compile with nvcc
#   3. a GPU is available

try:
    import pycuda.gpuarray as gpuarray
    import pycuda.autoinit
except ImportError:
    raise ImportError('PyCUDA is not installed. CUDA capabilities are not available.')

import pycuda.driver as driver
from pycuda.compiler import SourceModule

try:
    SourceModule('')
except driver.CompileError:
    raise ImportError('CUDA compiler not availble.')

if driver.Device.count() == 0:
    raise ImportError('No GPU is available.')

@array_stream
def csum(arrays, ignore_nan = False):
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
    if ignore_nan:
        arrays = map(np.nan_to_num, arrays)

    first = next(arrays)
    arr_gpu = gpuarray.to_gpu(first)

    for arr in arrays:
        arr_gpu += gpuarray.to_gpu(arr)

    return arr_gpu.get()

@array_stream
def caverage(arrays, weights = None, ignore_nan = False):
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
    
    first, fst_wgt = next(arrays), next(weights)
    arr_gpu = gpuarray.to_gpu(first)
    wgt_gpu = gpuarray.to_gpu(fst_wgt)
    for arr, wgt in zip(arrays, weights):
        arr_gpu += gpuarray.to_gpu(arr)
        wgt_gpu += gpuarray.to_gpu(wgt)
    
    arr_gpu /= wgt_gpu
    return arr_gpu.get()
    