# -*- coding: utf-8 -*-
"""
Flow controls
-------------
"""
from functools import partial
from glob import iglob

from .array_stream import ArrayStream
from .parallel import pmap, pmap_unordered


def iload(files, load_func, **kwargs):
    """
    Create a stream of arrays from files, which are loaded lazily.

    In cases where the consumer function is much faster than data loading,
    consider using :func:`pload` instead.

    Parameters
    ----------
    pattern : iterable of str or str
        Either an iterable of filenames or a glob-like pattern str.
    load_func : callable, optional
        Function taking a filename as its first arguments
    kwargs
        Keyword arguments are passed to ``load_func``.
    
    Yields
    ------
    arr: `~numpy.ndarray`
        Loaded data. 
    
    See Also
    --------
    pload : load files from parallel processes.
    
    Examples
    --------
    To load images using scikit-image ::

        from skimage.io import imread
        ims = iload('images_*.tif', imread)

    Keyword arguments are passed to the ``load_func``; for example, 
    to specify the scikit-image plugin ``'tifffile'``::

        ims = iload('images_*.tif', imread, plugin = 'tifffile')

    In case the list of images is already known::

        ims = iload(['im1.tif', 'im2.tif', 'im3.tif'], imread)
    """
    if isinstance(files, str):
        files = iglob(files)
    files = iter(files)

    yield from map(partial(load_func, **kwargs), files)


def pload(files, load_func, processes=1, **kwargs):
    """
    Create a stream of arrays from files, which are loaded lazily 
    from multiple processes. 
    
    This function should be preferred to :func:`iload` in cases where 
    the consumer function is much faster than the data can be loaded.

    Parameters
    ----------
    pattern : iterable of str or str
        Either an iterable of filenames or a glob-like pattern str.
    load_func : callable, optional
        Function taking a filename as its first arguments
    processes : int or None, optional
        Number of processes to use. If `None`, maximal number of processes
        is used. Default is one.
    kwargs
        Keyword arguments are passed to ``load_func``.

    Yields
    ------
    arr: `~numpy.ndarray`
        Loaded data. 

    See Also
    --------
    iload : load files lazily
    """
    if processes == 1:
        yield from iload(files, load_func, **kwargs)
        return

    if isinstance(files, str):
        files = iglob(files)
    files = iter(files)

    yield from pmap_unordered(partial(load_func, **kwargs), files, processes=processes)


# pmap does not support local functions
def _pipe(funcs, array):
    for func in funcs:
        array = func(array)
    return array


def ipipe(*args, **kwargs):
    """
    Pipe arrays through a sequence of functions. For example:

    ``pipe(f, g, h, stream)`` is equivalent to ::

        for arr in stream:
            yield f(g(h(arr)))
    
    Parameters
    ----------
    *funcs : callable
        Callable that support Numpy arrays in their first argument. These
        should *NOT* be generator functions.
    arrays : iterable
        Stream of arrays to be passed.
    processes : int or None, optional, keyword-only
        Number of processes to use. If `None`, maximal number of processes
        is used. Default is one.
    ntotal : int or None, optional, keyword-only
        If the length of `arrays` is known, but passing `arrays` as a list
        would take too much memory, the total number of arrays `ntotal` can be specified. This
        allows for `pmap` to chunk better in case of ``processes > 1``.
    
    Yields
    ------
    piped : ndarray
    """
    arrays = ArrayStream(args[-1])
    functions = tuple(reversed(args[:-1]))
    yield from pmap(partial(_pipe, functions), arrays, **kwargs)
