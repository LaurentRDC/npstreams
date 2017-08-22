# -*- coding: utf-8 -*-
"""
CUDA-accelerated streaming operations
-------------------------------------
"""

from . import array_stream, peek
import numpy as np
from warnings import warn

# Determine if 
#   1. pycuda is installed;
#   2. pycuda can compile with nvcc
#   3. a GPU is available

try:
    import pycuda.gpuarray as gpuarray
    import pycuda.autoinit
except ImportError:
    raise RuntimeError('PyCUDA is not installed. CUDA capabilities are not available.')

import pycuda.driver as driver
from pycuda.compiler import SourceModule

try:
    SourceModule('')
except driver.CompileError:
    raise RuntimeError('CUDA compiler not availble.')

if driver.Device.count() == 0:
    raise RuntimeError('No GPU is available.')

@array_stream
def csum(arrays, ignore_nan = False):
    """ 
    Streaming sum of array elements.

    Parameters
    ----------
    arrays : iterable
        Arrays to be summed. 
    ignore_nan : bool, optional
        If True, NaNs are ignored. Default is propagation of NaNs.
    
    Returns
    -------
    cuda_sum : ndarray
    """
    if ignore_nan:
        arrays = map(np.nan_to_num, arrays)

    first = next(arrays)
    arr_gpu = gpuarray.to_gpu(first)

    for arr in arrays:
        arr_gpu += gpuarray.to_gpu(arr)

    return arr_gpu.get()