# -*- coding: utf-8 -*-
""" 
Reliably benchmarking npstreams performance.
"""
import inspect
import sys
import timeit
from collections import namedtuple
from contextlib import redirect_stdout
from functools import partial
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
from numpy     import {func.__name__} as np_{func.__name__}
from npstreams import {func.__name__} as ns_{func.__name__}

np.random.seed(42056)

def stream():
    return (np.random.random({shape}) for _ in range(10))
"""

BenchmarkResults = namedtuple(
    "BenchmarkResults", field_names=["numpy_time", "npstreams_time", "shape"]
)


def autotimeit(statement, setup="pass", repeat=3):
    """ 
    Time a statement, automatically determining the number of times to
    run the statement so that the total excecution time is not too short. 

    .. versionadded:: 1.5.2
    
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
    """
    timer = timeit.Timer(stmt=statement, setup=setup)
    number, _ = timer.autorange()
    return min(timer.repeat(repeat=repeat, number=number)) / number


def benchmark(
    funcs=[np.average, np.mean, np.std, np.sum, np.prod],
    ufuncs=[np.add, np.multiply, np.power, np.true_divide, np.mod],
    shapes=[(4, 4), (8, 8), (16, 16), (64, 64)],
    file=None,
):
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
    file : file-like or None, optional
        File to which the benchmark results will be written. If None, sys.stdout will be used.
    """
    # Preliminaries
    console_width = min(get_terminal_size().columns, 80)
    func_test_name = "numpy.{f.__name__} vs npstreams.{f.__name__}".format
    ufunc_test_name = (
        "numpy.{f.__name__} vs npstreams.reduce_ufunc(numpy.{f.__name__}, ...)".format
    )

    # Determine justification based on maximal shape functions
    sh_just = max(map(lambda s: len(str(s)), shapes)) + 10

    # To make it easy to either write the results in a file or print to stdout,
    # We actually redirect stdout.
    if file is None:
        file = sys.stdout

    with redirect_stdout(file):
        # Start benchmarks --------------------------------------------------------
        print(
            "".ljust(console_width, "*"),
            "npstreams performance benchmark".upper().center(console_width),
            "",
            "    npstreams".ljust(15) + f" {__version__}",
            "    NumPy".ljust(15) + f" {np.__version__}",
            "",
            "    Speedup is NumPy time divided by npstreams time (Higher is better)",
            "".ljust(console_width, "*"),
            sep="\n",
        )

        # Determine valid ufuncs and funcs first ----------------------------------
        valid_ufuncs = comparable_ufuncs(ufuncs, file)
        valid_funcs = comparable_funcs(funcs, file)

        # Benchmarking functions --------------------------------------------------
        for func in sorted(valid_funcs, key=lambda fn: fn.__name__):
            print(func_test_name(f=func).center(console_width), "\n")

            for (np_time, ns_time, shape) in benchmark_func(func, shapes):
                speedup = np_time / ns_time
                print(
                    "    ",
                    f"shape = {shape}".ljust(sh_just),
                    f"speedup = {speedup:.4f}x",
                )

            print("".ljust(console_width, "-"))

        # Benchmarking universal functions ----------------------------------------
        for ufunc in sorted(valid_ufuncs, key=lambda fn: fn.__name__):
            print(ufunc_test_name(f=ufunc).center(console_width), "\n")

            for (np_time, ns_time, shape) in benchmark_ufunc(ufunc, shapes):
                speedup = np_time / ns_time
                print(
                    "    ",
                    f"shape = {shape}".ljust(sh_just),
                    f"speedup = {speedup:.4f}x",
                )

            print("".ljust(console_width, "-"))


def benchmark_ufunc(ufunc, shapes):
    """ 
    Compare the running time between a NumPy ufunc and the npstreams equivalent.
    
    Parameters
    ----------
    ufunc : NumPy ufunc

    shapes : iterable of tuples, optional
        Shapes of arrays to test. Streams of random numbers will be generated with arrays of those shapes.
        The sequence lengths are fixed.
    
    Yields
    ------
    results : BenchmarkResults
    """
    for shape in shapes:

        numpy_statement = f"{ufunc.__name__}.reduce(stack(stream()), axis = -1)"
        npstreams_statement = f"reduce_ufunc(stream(), {ufunc.__name__}, axis = -1)"

        with np.errstate(invalid="ignore"):
            np_time = autotimeit(
                numpy_statement, UFUNC_SETUP.format(ufunc=ufunc, shape=shape)
            )
            ns_time = autotimeit(
                npstreams_statement, UFUNC_SETUP.format(ufunc=ufunc, shape=shape)
            )

        yield BenchmarkResults(np_time, ns_time, shape)


def benchmark_func(func, shapes):
    """ 
    Compare the running time between a NumPy func and the npstreams equivalent.
    
    Parameters
    ----------
    func : NumPy func

    shapes : iterable of tuples, optional
        Shapes of arrays to test. Streams of random numbers will be generated with arrays of those shapes.
        The sequence lengths are fixed.
    
    Yields
    ------
    results : BenchmarkResults
    """
    for shape in shapes:

        numpy_statement = f"np_{func.__name__}(stack(stream()), axis = -1)"
        npstreams_statement = f"ns_{func.__name__}(stream(), axis = -1)"

        with np.errstate(invalid="ignore"):
            np_time = autotimeit(
                numpy_statement, FUNC_SETUP.format(func=func, shape=shape)
            )
            ns_time = autotimeit(
                npstreams_statement, FUNC_SETUP.format(func=func, shape=shape)
            )

        yield BenchmarkResults(np_time, ns_time, shape)


def comparable_ufuncs(ufuncs, file):
    """ 
    Yields ufuncs that can be compared between numpy and npstreams. 
    
    Parameters
    ----------
    ufuncs : iterable of NumPy ufunc
        NumPy ufuncs to check. Ufuncs that cannot be compared will be skipped.

    Yields
    ------
    ufunc : callable
        NumPy ufuncs that can be compared with npstreams. 
    """
    for ufunc in ufuncs:
        if not isinstance(ufunc, np.ufunc):
            print(
                f"Skipping function {ufunc.__name__} as it is not a NumPy Universal Function"
            )
            continue

        try:
            _check_binary_ufunc(ufunc)
        except ValueError:
            print(
                f"Skipping function {ufunc.__name__} as it is not a valid binary ufunc"
            )
        else:
            yield ufunc


def comparable_funcs(funcs, file):
    """ 
    Yields NumPy functions that have npstreams equivalents. 
    
    Parameters
    ----------
    ufuncs : iterable of NumPy functions
        NumPy funcs to check.

    Yields
    ------
    ufunc : callable
        NumPy funcs that have npstreams equivalents. 
    """
    import npstreams

    npstreams_functions = set(
        name for name, value in inspect.getmembers(npstreams, inspect.isfunction)
    )
    for func in funcs:
        if func.__name__ not in npstreams_functions:
            print(
                f"Skipping function {func.__name__} as there is no npstreams equivalent"
            )
        else:
            yield func
