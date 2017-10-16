# -*- coding: utf-8 -*-
"""
Iterator/Generator utilities 
----------------------------
"""
from collections import deque
from functools import wraps
from itertools import islice, chain, tee

def chunked(iterable, chunksize = 1):
    """
    Generator yielding multiple iterables of length 'chunksize'.

    Parameters
    ----------
    iterable : iterable
        Iterable to be chunked. 
    chunksize : int, optional
        Chunk size. 

    Yields
    ------
    chunk : iterable
        Iterable of size `chunksize`. In special case of iterable not being
        divisible by `chunksize`, the last `chunk` might be smaller.
    """
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

def itercopy(iterable, copies = 2):
    """
    Split iterable into 'copies'. Once this is done, the original iterable should
    not be used again.

    Parameters
    ----------
    iterable : iterable
        Iterable to be split. Once it is split, the original iterable
        should not be used again.
    copies : int, optional
        Number of copies. Also determines the number of returned iterables
    
    Returns
    -------
    iter1, iter2, ... : iterable
        Copies of ``iterable``.
    
    Examples
    --------
    By rebinding the name of the original iterable, we make sure that it
    will never be used again.

    >>> from itertools import count
    >>> from npstreams import itercopy
    >>> evens = (2*n for n in count(0))
    >>> evens, evens_copy = itercopy(evens, copies = 2)

    See Also
    --------
    itertools.tee : equivalent function
    """
    # itercopy is included because documentation of itertools.tee isn't obvious
    # to everyone
    return tee(iterable, copies)

def linspace(start, stop, num, endpoint = True):
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

    step = (stop - start)/num

    val = start
    for _ in range(num):
        yield val
        val += step

    if endpoint:
        yield stop

def multilinspace(start, stop, num, endpoint = True):
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
    >>> multispace = multilinspaces(start = (0, 0), stop = (1, 1), num = 4, endpoint = False)
    >>> print(list(multispace))
    [(0, 0), (0.25, 0.25), (0.5, 0.5), (0.75, 0.75)]

    See also
    --------
    linspace : generate a linear space between two numbers
    """	
    start, stop = tuple(start), tuple(stop)
    if len(start) != len(stop):
        raise ValueError('start and stop must have the same length')

    spaces = tuple(linspace(a, b, num = num, endpoint = endpoint) for a, b in zip(start, stop))
    yield from zip(*spaces)

def last(stream):
    """ 
    Retrieve the last item from a stream/iterator, consuming 
    iterables in the process. If empty stream, returns None. 
    """
    # Wonderful idea from itertools recipes
    # https://docs.python.org/3.6/library/itertools.html#itertools-recipes
    try:
        return deque(stream, maxlen = 1)[0]
    except IndexError:	# Empty stream
        return None

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