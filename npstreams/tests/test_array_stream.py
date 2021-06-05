# -*- coding: utf-8 -*-
import numpy as np

from npstreams.array_stream import array_stream, ArrayStream


@array_stream
def iden(arrays):
    yield from arrays


def test_array_stream_decorator_type():
    """Test that all object from an array stream are ndarrays"""

    stream = [0, 1, np.array([1])]
    for arr in iden(stream):
        assert isinstance(arr, np.ndarray)


def test_single_array():
    """Test that a 'stream' consisting of a single array is repackaged into an iterable"""
    stream = np.array([1, 2, 3])
    assert len(list(iden(stream))) == 1


def test_array_stream_length_hint_sized_iterable():
    """Test the accuracy of __length_hint__ for ArrayStream constructed
    from a sized iterable"""
    iterable = [1, 2, 3, 4, 5]
    a = ArrayStream(iterable)
    assert len(iterable) == a.__length_hint__()


def test_array_stream_length_hint_not_sized_iterable():
    """Test that __length_hint__ returns NotImplemented for ArrayStream constructed
    from an unsized iterable"""
    iterable = (0 for _ in range(10))
    a = ArrayStream(iterable)
    assert a.__length_hint__() is NotImplemented


def test_array_stream_conversion_to_array():
    """Test that numpy.array(Arraystream(...)) returns an array built as a stack of arrays"""
    a = ArrayStream([np.random.random((16, 16)) for _ in range(10)])
    arr = np.array(a)
    assert arr.shape == (16, 16, 10)
