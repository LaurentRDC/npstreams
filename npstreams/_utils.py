# -*- coding: utf-8 -*-
"""
Utilities
---------
"""
import numpy as np

def array_stream(func):
    """ Decorates streaming functions to make sure that the stream
    is a stream of ndarrays. If the stream is in fact a single ndarray, 
    this ndarray is repackaged into a sequence """
    def decorated(arrays, *args, **kwargs):
        if isinstance(arrays, np.ndarray):
            arrays = (arrays,)
        return func(map(np.asarray, arrays), *args, **kwargs)
    return decorated