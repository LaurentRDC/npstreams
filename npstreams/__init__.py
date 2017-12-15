# -*- coding: utf-8 -*-
__author__ = 'Laurent P. René de Cotret'
__email__ = 'laurent.renedecotret@mail.mcgill.ca'
__license__ = 'BSD'
__version__ = '1.2.0' # TODO: automatic versioning?

# Order of import is important
# because of inter-dependency

from .array_stream import array_stream, ipipe, iload
from .array_utils import nan_to_num
from .linalg import idot, itensordot, ieinsum, iinner
from .parallel import pmap, pmap_unordered, preduce
from .iter_utils import cyclic, last, chunked, multilinspace, linspace, peek, itercopy, primed
from .reduce import ireduce_ufunc, preduce_ufunc, reduce_ufunc
from .stacking import istack, iflatten
from .stats import iaverage, imean, inanmean, istd, inanstd, isem, ivar, inanvar, ihistogram
from .numerics import isum, inansum, iprod, inanprod, isub, iall, iany, imax, imin
