# -*- coding: utf-8 -*-
""" 
Reliably benchmarking npstreams performance.
"""

import inspect
import timeit
from shutil import get_terminal_size
import numpy as np

from . import __version__
from .reduce import _check_binary_ufunc


UFUNC_SETUP = """
from npstreams import reduce_ufunc, stack
import numpy as np
from numpy import {ufunc.__name__}

np.random.seed(42056)

def stream():
    return (np.random.random({shape}) for _ in range(10))
"""

FUNC_SETUP = """
from npstreams import stack
import numpy as np
from numpy import {func.__name__} as np_{func.__name__}
from npstreams import {func.__name__} as ns_{func.__name__}

np.random.seed(42056)

def stream():
    return (np.random.random({shape}) for _ in range(10))
"""

def autotimeit(statement, setup = 'pass', repeat = 3):
    """ 
    Time a statement, automatically determining the number of times to
    run the statement so that the total excecution time is not too short. 
    
    Parameters
    ----------
    statement: string
        Statement to time. The statement will be executed after the `setup` statement.
    setup : string, optional
        Setup statement executed before timing starts. 
    repeat : int, optional
        Number of repeated timing to execute.
    
    Returns
    -------
    time : float
        Minimal time per execution of `statement` [seconds].
    
    .. versionadded:: 1.5.2
    """
    timer = timeit.Timer(stmt = statement, setup = setup)
    number, _ = timer.autorange()
    return min(timer.repeat(repeat = repeat, number = number))/number

def benchmark(funcs =  [np.average, np.mean, np.std, np.sum, np.prod], 
              ufuncs = [np.add, np.multiply, np.power, np.true_divide, np.mod], 
              shapes = [(4,4), (8,8), (16,16), (64,64)]):
    """ 
    Benchmark npstreams against numpy and print the results.

    There are two categories of benchmarks. The first category compares NumPy functions against
    npstreams versions of the same functions. The second category compares NumPy universal functions
    against dynamically-generated npstreams versions of those same universal functions.

    All benchmarks compare a reduction operation on a stream of arrays of varying sizes. The sequence length is fixed.
    
    .. versionadded:: 1.5.2
    
    Parameters
    ----------
    funcs : iterable of NumPy functions, optional
        NumPy functions to compare. An equivalent must exist in npstreams, e.g. `np.average` and `npstreams.average` .
        Functions without equivalents will be skipped.
    ufuncs : iterable of NumPy ufunc, optional
        Invalid ufuncs (e.g. non-binary ufuncs) will be skipped.
    shapes : iterable of tuples, optional
        Shapes of arrays to test. Streams of random numbers will be generated with arrays of those shapes.
        The sequence lengths are fixed.
    """
    # Preliminaries
    console_width = min(get_terminal_size().columns, 80)
    func_test_name = 'np.{f.__name__} vs ns.{f.__name__}'.format
    ufunc_test_name = 'np.{f.__name__} vs ns.reduce_ufunc(np.{f.__name__}, ...)'.format

    # Determine justification based on maximal shape functions
    sh_just = max(map(lambda s : len(str(s)), shapes)) + 10

    # Determine valid ufuncs and funcs first ----------------------------------
    valid_ufuncs = list()
    for ufunc in ufuncs:
        if not isinstance(ufunc, np.ufunc):
            print('Skipping function {func} as it is not a NumPy Universal Function'.format(func = ufunc.__name__))
            continue
        try:
            _check_binary_ufunc(ufunc)
        except ValueError:
            print('Skipping function {func} as it is not a valid binary ufunc'.format(func = ufunc.__name__))
            continue
        # If we got here, the ufunc is a valid binary universal function
        valid_ufuncs.append(ufunc)
    
    valid_funcs = list()
    import npstreams
    npstreams_functions = set(name for name, value in inspect.getmembers(npstreams, inspect.isfunction))
    for func in funcs:
        if func.__name__ not in npstreams_functions:
            print('Skipping function {func} as there is no npstreams equivalent'.format(func = ufunc.__name__))
            continue
        else:
            valid_funcs.append(func)
        
    # Start benchmarks --------------------------------------------------------
    print('')
    print(''.ljust(console_width, '*'))
    print('npstreams performance benchmark'.center(console_width))
    print("    npstreams".ljust(15), "{}".format(__version__))
    print("    NumPy".ljust(15), "{}".format(np.__version__))
    print("    Speedup is NumPy time divided by npstreams time (Higher is better)")
    print(''.ljust(console_width, '*'))

    # Benchmarking functions --------------------------------------------------
    for func in sorted(valid_funcs, key = lambda fn: fn.__name__):
        print(func_test_name(f = func).center(console_width))
        for shape in shapes:

            numpy_statement     = 'np_{}(stack(stream()), axis = -1)'.format(func.__name__)
            npstreams_statement = 'ns_{}(stream(), axis = -1)'.format(func.__name__)

            with np.errstate(invalid = 'ignore'):
                np_time = autotimeit(numpy_statement, FUNC_SETUP.format(func = func, shape = shape))
                ns_time = autotimeit(npstreams_statement, FUNC_SETUP.format(func = func, shape = shape))

            print("    ", "shape = {}".format(shape).ljust(sh_just), "speedup = {:.4f}x".format(np_time / ns_time))

        print(''.ljust(console_width, '-'))

    # Benchmarking universal functions ----------------------------------------
    for ufunc in sorted(valid_ufuncs, key = lambda fn: fn.__name__):
        print(ufunc_test_name(f = ufunc).center(console_width))
        for shape in shapes:
            
            numpy_statement     = '{ufunc}.reduce(stack(stream()), axis = -1)'.format(ufunc = ufunc.__name__)
            npstreams_statement = 'reduce_ufunc(stream(), {ufunc}, axis = -1)'.format(ufunc = ufunc.__name__)

            with np.errstate(invalid = 'ignore'):
                np_time = autotimeit(numpy_statement, UFUNC_SETUP.format(ufunc = ufunc, shape = shape))
                ns_time = autotimeit(npstreams_statement, UFUNC_SETUP.format(ufunc = ufunc, shape = shape))
            
            print("    ", "shape = {}".format(shape).ljust(sh_just), "speedup = {:.4f}x".format(np_time / ns_time))
        
        print(''.ljust(console_width, '-'))
