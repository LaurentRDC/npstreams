# -*- coding: utf-8 -*-
__author__ = 'Laurent P. René de Cotret'
__email__ = 'laurent.renedecotret@mail.mcgill.ca'
__license__ = 'BSD'
__version__ = '0.1.3' # TODO: automatic versioning?

from numpy import array, isnan
# TODO: is in-place justified?
def _nan_to_num(arr, fill):
    """ Replace NaNs in `array` with `fill`. Keyword-arguments
    are passed to numpy.nan_to_num"""
    with_nans = array(arr)
    with_nans[isnan(with_nans)] = fill
    return with_nans

from .parallel import pmap, preduce
from .iter_utils import last, chunked, multilinspace, linspace, peek
from .reduce import stream_reduce
from .stats import iaverage, imean, inanmean, istd, inanstd, isem, ivar, inanvar
from .numerics import isum, inansum, psum, iprod, pprod, inanprod