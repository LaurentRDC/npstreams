# -*- coding: utf-8 -*-
"""
Array utilities 
---------------
"""
import numpy as np


def nan_to_num(array, fill_value=0.0, copy=True):
    """
    Replace NaNs with another fill value. 

    Parameters
    ----------
    array : array_like
        Input data.
    fill_value : float, optional
        NaNs will be replaced by ``fill_value``. Default is 0.0, in keeping
        with ``numpy.nan_to_num``.
    copy : bool, optional
        Whether to create a copy of `array` (True) or to replace values
        in-place (False). The in-place operation only occurs if
        casting to an array does not require a copy.
    
    Returns
    -------
    out : ndarray
        Array without NaNs. If ``array`` was not of floating or complearray type,
        ``array`` is returned unchanged.
    
    Notes
    -----
    Contrary to ``numpy.nan_to_num``, this functions does not handle
    infinite values.

    See Also
    --------
    numpy.nan_to_num : replace NaNs and Infs with zeroes.
    """
    array = np.array(array, subok=True, copy=copy)
    dtype = array.dtype.type

    # Non-inexact types do not have NaNs
    if not np.issubdtype(dtype, np.inexact):
        return array

    iscomplex = np.issubdtype(dtype, np.complexfloating)
    dest = (array.real, array.imag) if iscomplex else (array,)
    for d in dest:
        np.copyto(d, fill_value, where=np.isnan(d))
    return array
