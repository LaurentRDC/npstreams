# -*- coding: utf-8 -*-
__author__ = 'Laurent P. René de Cotret'
__email__ = 'laurent.renedecotret@mail.mcgill.ca'
__license__ = 'BSD'
__version__ = '0.1.4' # TODO: automatic versioning?

from .iter_utils import chunked, last, linspace, multilinspace, peek
from .numerics import inanprod, inansum, iprod, isub, isum, pprod, psum
from .parallel import pmap, preduce
from .reduce import iall, iany, stream_reduce
from .stacking import istack
from .stats import (iaverage, imean, inanmean, inanstd, inanvar, isem, istd,
                    ivar)
