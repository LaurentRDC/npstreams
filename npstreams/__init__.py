# -*- coding: utf-8 -*-
__author__ = 'Laurent P. René de Cotret'
__email__ = 'laurent.renedecotret@mail.mcgill.ca'
__license__ = 'BSD'
__version__ = '0.2' # TODO: automatic versioning?

from .array_stream import array_stream, ipipe

from .linalg import idot, itensordot, ieinsum
from .parallel import pmap, preduce
from .iter_utils import last, chunked, multilinspace, linspace, peek
from .reduce import stream_reduce
from .stacking import istack, iflatten
from .stats import iaverage, imean, inanmean, istd, inanstd, isem, ivar, inanvar
from .numerics import isum, inansum, psum, iprod, pprod, inanprod, isub, iall, iany