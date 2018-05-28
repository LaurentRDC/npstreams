# -*- coding: utf-8 -*-
"""
Parallelization utilities 
-------------------------
"""
from collections.abc import Sized
from functools import partial, reduce
from multiprocessing import Pool

from .iter_utils import chunked


def preduce(func, iterable, args = None, kwargs = None, processes = 1, ntotal = None):
    """
    Parallel application of the reduce function, with keyword arguments.

    Parameters
    ----------
    func : callable
        Function to be applied to every element of `iterable`.
    iterable : iterable
        Iterable of items to be reduced. Generators are consumed.
    args : tuple or None, optional
        Positional arguments of `function`.
    kwargs : dictionary or None, optional
        Keyword arguments of `function`.
    processes : int or None, optional
        Number of processes to use. If `None`, maximal number of processes
        is used. Default is one.
    ntotal : int or None, optional
        If the length of `iterable` is known, but passing `iterable` as a list
        would take too much memory, the total length `ntotal` can be specified. This
        allows for `preduce` to chunk better.

    Returns
    -------
    reduced : object

    Notes
    -----
    If `processes` is 1, `preduce` is equivalent to functools.reduce with the
    added benefit of using `args` and `kwargs`, but `initializer` is not supported.
    """
    if kwargs is None:
        kwargs = dict()
    
    if args is None:
        args = tuple()

    func = partial(func, *args, **kwargs)

    if processes == 1:
        return reduce(func, iterable)

    with Pool(processes) as pool:
        chunksize = 1
        if isinstance(iterable, Sized):
            chunksize = max(1, int(len(iterable)/pool._processes))
        elif ntotal is not None:
            chunksize = max(1, int(ntotal/pool._processes))
        
        # Some reductions are order-sensitive
        res = pool.imap(partial(reduce, func), tuple(chunked(iterable, chunksize)))
        return reduce(func, res)

def pmap(func, iterable, args = None, kwargs = None, processes = 1, ntotal = None):
    """
    Parallel application of a function with keyword arguments.

    Parameters
    ----------
    func : callable
        Function to be applied to every element of `iterable`.
    iterable : iterable
        Iterable of items to be mapped.
    args : tuple or None, optional
        Positional arguments of `function`.
    kwargs : dictionary or None, optional
        Keyword arguments of `function`.
    processes : int or None, optional
        Number of processes to use. If `None`, maximal number of processes
        is used. Default is one.
    ntotal : int or None, optional
        If the length of `iterable` is known, but passing `iterable` as a list
        would take too much memory, the total length `ntotal` can be specified. This
        allows for `pmap` to chunk better.

    Yields
    ------
    Mapped values.

    See Also
    --------
    pmap_unordered : parallel map that does not preserve order

    Notes
    -----
    If `processes` is 1, `pmap` reduces to `map`, with the added benefit of
    of using `kwargs`
    """
    if kwargs is None:
        kwargs = dict()
    
    if args is None:
        args = tuple()

    func = partial(func, *args, **kwargs)

    if processes == 1:
        yield from map(func, iterable)
        return
    
    with Pool(processes) as pool:
        chunksize = 1
        if isinstance(iterable, Sized):
            chunksize = max(1, int(len(iterable)/pool._processes))
        elif ntotal is not None:
            chunksize = max(1, int(ntotal/pool._processes))

        yield from pool.imap(func = func, iterable = iterable, chunksize = chunksize)

def pmap_unordered(func, iterable, args = None, kwargs = None, processes = 1, ntotal = None):
    """
    Parallel application of a function with keyword arguments in no particular order. 
    This can reduce memory usage because results are not accumulated so that the order is preserved.

    Parameters
    ----------
    func : callable
        Function to be applied to every element of `iterable`.
    iterable : iterable
        Iterable of items to be mapped.
    args : tuple or None, optional
        Positional arguments of `function`.
    kwargs : dictionary or None, optional
        Keyword arguments of `function`.
    processes : int or None, optional
        Number of processes to use. If `None`, maximal number of processes
        is used. Default is one.
    ntotal : int or None, optional
        If the length of `iterable` is known, but passing `iterable` as a list
        would take too much memory, the total length `ntotal` can be specified. This
        allows for `pmap` to chunk better.

    Yields
    ------
    Mapped values.

    See Also
    --------
    pmap : parallel map that preserves order

    Notes
    -----
    If `processes` is 1, `pmap_unordered` reduces to `map`, with the added benefit of
    of using `kwargs`
    """
    if kwargs is None:
        kwargs = dict()
    
    if args is None:
        args = tuple()
    
    func = partial(func, *args, **kwargs)

    if processes == 1:
        yield from map(func, iterable)
        return
    
    with Pool(processes) as pool:
        chunksize = 1
        if isinstance(iterable, Sized):
            chunksize = max(1, int(len(iterable)/pool._processes))
        elif ntotal is not None:
            chunksize = max(1, int(ntotal/pool._processes))

        yield from pool.imap_unordered(func = func, iterable = iterable, chunksize = chunksize)
