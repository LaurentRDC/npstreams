# -*- coding: utf-8 -*-
__author__ = "Laurent P. René de Cotret"
__email__ = "laurent.renedecotret@mail.mcgill.ca"
__license__ = "BSD"
__version__ = "1.6.2"

# Order of import is important
# because of inter-dependency
from .utils import deprecated

from .benchmarks import benchmark
from .array_stream import array_stream, ArrayStream
from .array_utils import nan_to_num
from .linalg import idot, itensordot, ieinsum, iinner
from .parallel import pmap, pmap_unordered, preduce
from .flow import ipipe, iload, pload
from .iter_utils import (
    cyclic,
    last,
    chunked,
    multilinspace,
    linspace,
    peek,
    itercopy,
    primed,
    length_hint,
)
from .reduce import ireduce_ufunc, preduce_ufunc, reduce_ufunc
from .stacking import stack
from .stats import (
    iaverage,
    average,
    imean,
    mean,
    istd,
    std,
    ivar,
    var,
    isem,
    sem,
    average_and_var,
    ihistogram,
)
from .numerics import isum, sum, iprod, prod, isub, iall, iany, imax, imin
