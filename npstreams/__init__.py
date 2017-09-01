# -*- coding: utf-8 -*-
__author__ = 'Laurent P. René de Cotret'
__email__ = 'laurent.renedecotret@mail.mcgill.ca'
__license__ = 'BSD'
__version__ = '0.5' # TODO: automatic versioning?

from .array_stream import array_stream, ipipe

from .linalg import idot, itensordot, ieinsum, iinner
from .parallel import pmap, pmap_unordered, preduce
from .iter_utils import last, chunked, multilinspace, linspace, peek, itercopy, primed
from .reduce import ireduce_ufunc, reduce_ufunc
from .stacking import istack, iflatten
from .stats import iaverage, imean, inanmean, istd, inanstd, isem, ivar, inanvar
from .numerics import isum, inansum, iprod, inanprod, isub, iall, iany, imax, imin