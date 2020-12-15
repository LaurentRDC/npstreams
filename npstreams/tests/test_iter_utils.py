# -*- coding: utf-8 -*-

from itertools import repeat
from npstreams import last, chunked, linspace, multilinspace, cyclic, length_hint
import pytest


def test_last_trivial():
    """ Test last() on iterable of identical values """
    i = repeat(1, 10)
    assert last(i) == 1


def test_last_on_empty_iterable():
    """ Test that last() raises RuntimeError for empty iterable """
    with pytest.raises(RuntimeError):
        last(list())


def test_cyclic_numbers():
    """ """
    permutations = set(cyclic((1, 2, 3)))
    assert (1, 2, 3) in permutations
    assert (2, 3, 1) in permutations
    assert (3, 1, 2) in permutations
    assert len(permutations) == 3


def test_linspace_endpoint():
    """ Test that the endpoint is included by linspace() when appropriate"""
    space = linspace(0, 1, num=10, endpoint=True)
    assert last(space) == 1

    space = linspace(0, 1, num=10, endpoint=False)
    assert round(abs(last(space) - 0.9), 7) == 0


def test_linspace_length():
    """ Test that linspace() returns an iterable of the correct length """
    space = list(linspace(0, 1, num=13, endpoint=True))
    assert len(space) == 13

    space = list(linspace(0, 1, num=13, endpoint=False))
    assert len(space) == 13


def test_multilinspace_endpoint():
    """ Test that the endpoint is included by linspace() when appropriate"""
    space = multilinspace((0, 0), (1, 1), num=10, endpoint=True)
    assert last(space) == (1, 1)

    space = multilinspace((0, 0), (1, 1), num=10, endpoint=False)
    # Unfortunately there is no assertSequenceAlmostEqual
    assert last(space) == (0.8999999999999999, 0.8999999999999999)


def test_multilinspace_length():
    """ Test that linspace() returns an iterable of the correct length """
    space = list(multilinspace((0, 0), (1, 1), num=13, endpoint=True))
    assert len(space) == 13

    space = list(multilinspace((0, 0), (1, 1), num=13, endpoint=False))
    assert len(space) == 13


def test_chunked_larger_chunksize():
    """ Test chunked() with a chunksize larger that the iterable it """
    i = repeat(1, 10)
    chunks = chunked(i, chunksize=15)
    assert len(list(chunks)) == 1  # One single chunk is returned


def test_chunked_on_infinite_generator():
    """ Test chunked() on an infinite iterable """
    i = repeat(1)
    chunks = chunked(i, chunksize=15)
    for _ in range(10):
        assert len(next(chunks)) == 15


def test_chunked_chunked_nonint_chunksize():
    """ Test that chunked raises a TypeError immediately if `chunksize` is not an integer """
    with pytest.raises(TypeError):
        i = repeat(1)
        chunks = chunked(i, chunksize=15.0)


def test_length_hint_on_sized():
    """ Test length_hint on a sized iterable """
    l = [1, 2, 3, 4, 5]
    assert length_hint(l) == len(l)


def test_length_hint_on_unsized():
    """ Test length_hint on an unsized iterable returns the default """
    l = (0 for _ in range(10))
    assert length_hint(l, default=0) == 0


def test_length_hint_on_method_if_implemented():
    """ Test length_hint returns the same as __length_hint__ if implemented """

    class WithHint:
        """ Some dummy class with a length hint """

        def __length_hint__(self):
            return 1

    assert length_hint(WithHint(), default=0) == 1
