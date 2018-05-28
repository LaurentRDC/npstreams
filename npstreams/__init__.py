# -*- coding: utf-8 -*-
__author__ = 'Laurent P. René de Cotret'
__email__ = 'laurent.renedecotret@mail.mcgill.ca'
__license__ = 'BSD'
__version__ = '1.5.1'

# Order of import is important
# because of inter-dependency
from .utils import deprecated

from .array_stream import array_stream, ipipe, iload, pload
from .array_utils import nan_to_num
from .linalg import idot, itensordot, ieinsum, iinner
from .parallel import pmap, pmap_unordered, preduce
from .iter_utils import cyclic, last, chunked, multilinspace, linspace, peek, itercopy, primed
from .reduce import ireduce_ufunc, preduce_ufunc, reduce_ufunc
from .stacking import stack
from .stats import (iaverage, average, imean, mean, istd, std, 
                    ivar, var, isem, sem, ihistogram)
from .numerics import isum, sum, iprod, prod, isub, iall, iany, imax, imin
