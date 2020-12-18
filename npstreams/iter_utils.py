# -*- coding: utf-8 -*-
"""
Iterator/Generator utilities 
----------------------------
"""
from collections import deque
from functools import wraps
from itertools import chain, islice, tee


def primed(gen):
    """
    Decorator that primes a generator function, i.e. runs the function
    until the first ``yield`` statement. Useful in cases where there
    are preliminary checks when creating the generator.
    """

    @wraps(gen)
    def primed_gen(*args, **kwargs):
        generator = gen(*args, **kwargs)
        next(generator)
        return generator

    return primed_gen


@primed
def chunked(iterable, chunksize):
    """
    Generator yielding multiple iterables of length 'chunksize'.

    Parameters
    ----------
    iterable : iterable
        Iterable to be chunked.
    chunksize : int
        Chunk size.

    Yields
    ------
    chunk : iterable
        Iterable of size `chunksize`. In special case of iterable not being
        divisible by `chunksize`, the last `chunk` will be smaller.

    Raises
    ------
    TypeError : if `chunksize` is not an integer.
    """
    if not isinstance(chunksize, int):
        raise TypeError(
            f"Expected `chunksize` to be an integer, but received {chunksize}"
        )

    yield

    iterable = iter(iterable)

    next_chunk = tuple(islice(iterable, chunksize))
    while next_chunk:
        yield next_chunk
        next_chunk = tuple(islice(iterable, chunksize))


def peek(iterable):
    """
    Peek ahead in an iterable.

    Parameters
    ----------
    iterable : iterable

    Returns
    -------
    first : object
        First element of ``iterable``
    stream : iterable
        Iterable containing ``first`` and all other elements from ``iterable``
    """
    iterable = iter(iterable)
    ahead = next(iterable)
    return ahead, chain([ahead], iterable)


def itercopy(iterable, copies=2):
    """
    Split iterable into 'copies'. Once this is done, the original iterable *should
    not* be used again.

    Parameters
    ----------
    iterable : iterable
        Iterable to be split. Once it is split, the original iterable
        should not be used again.
    copies : int, optional
        Number of copies. Also determines the number of returned iterables.

    Returns
    -------
    iter1, iter2, ... : iterable
        Copies of ``iterable``.

    Examples
    --------
    By rebinding the name of the original iterable, we make sure that it
    will never be used again.

    >>> from npstreams import itercopy
    >>> evens = (2*n for n in range(1000))
    >>> evens, evens_copy = itercopy(evens, copies = 2)

    See Also
    --------
    itertools.tee : equivalent function
    """
    # itercopy is included because documentation of itertools.tee isn't obvious
    # to everyone
    return tee(iterable, copies)


def linspace(start, stop, num, endpoint=True):
    """
    Generate linear space. This is sometimes more appropriate than
    using `range`.

    Parameters
    ----------
    start : float
        The starting value of the sequence.
    stop : float
        The end value of the sequence.
    num : int
        Number of samples to generate.
    endpoint : bool, optional
        If True (default), the endpoint is included in the linear space.

    Yields
    ------
    val : float

    See also
    --------
    numpy.linspace : generate linear space as a dense array.
    """
    # If endpoint are to be counted in,
    # step does not count the last yield
    if endpoint:
        num -= 1

    step = (stop - start) / num

    val = start
    for _ in range(num):
        yield val
        val += step

    if endpoint:
        yield stop


def multilinspace(start, stop, num, endpoint=True):
    """
    Generate multilinear space, for joining the values in two iterables.

    Parameters
    ----------
    start : iterable of floats
        The starting value. This iterable will be consumed.
    stop : iterable of floats
        The end value. This iterable will be consumed.
    num : int
        Number of samples to generate.
    endpoint : bool, optional
        If True (default), the endpoint is included in the linear space.

    Yields
    ------
    val : tuple
        Tuple of the same length as start and stop

    Examples
    --------
    >>> from npstreams import multilinspace
    >>> multispace = multilinspace(start = (0, 0), stop = (1, 1), num = 4, endpoint = False)
    >>> print(list(multispace))
    [(0, 0), (0.25, 0.25), (0.5, 0.5), (0.75, 0.75)]

    See also
    --------
    linspace : generate a linear space between two numbers
    """
    start, stop = tuple(start), tuple(stop)
    if len(start) != len(stop):
        raise ValueError("start and stop must have the same length")

    spaces = tuple(
        linspace(a, b, num=num, endpoint=endpoint) for a, b in zip(start, stop)
    )
    yield from zip(*spaces)


def last(stream):
    """
    Retrieve the last item from a stream/iterator, consuming
    iterables in the process. If empty stream, a RuntimeError is raised.
    """
    # Wonderful idea from itertools recipes
    # https://docs.python.org/3.9/library/itertools.html#itertools-recipes
    try:
        return deque(stream, maxlen=1)[0]
    except IndexError:
        raise RuntimeError("Empty stream")


def cyclic(iterable):
    """
    Yields cyclic permutations of an iterable.

    Examples
    --------
    >>> from npstreams import cyclic
    >>> list(cyclic((1,2,3)))
    [(1, 2, 3), (3, 1, 2), (2, 3, 1)]
    """
    iterable = tuple(iterable)
    n = len(iterable)
    yield from (tuple(iterable[i - j] for i in range(n)) for j in range(n))


def length_hint(obj, default=0):
    """
    Return an estimate of the number of items in ``obj``.

    This is useful for presizing containers when building from an
    iterable.

    If the object supports len(), the result will be
    exact. Otherwise, it may over- or under-estimate by an
    arbitrary amount. The result will be an integer >= 0.

    Notes
    -----
    Source : https://www.python.org/dev/peps/pep-0424/

    Examples
    --------
    >>> from npstreams import length_hint
    >>> length_hint([1,2,3,4,5])          # Should be exact
    5
    >>> length_hint(None, default = 15)   # Does not implement __length_hint__
    15
    """
    try:
        return len(obj)
    except TypeError:
        try:
            get_hint = type(obj).__length_hint__
        except AttributeError:
            return default
        try:
            hint = get_hint(obj)
        except TypeError:
            return default
        if hint is NotImplemented:
            return default
        if not isinstance(hint, int):
            raise TypeError("Length hint must be an integer, not %r" % type(hint))
        if hint < 0:
            raise ValueError("__length_hint__() should return >= 0")
        return hint
