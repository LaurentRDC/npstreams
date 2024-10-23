# -*- coding: utf-8 -*-

from collections.abc import Iterator
from functools import wraps

import numpy as np
from numpy import asanyarray

from .iter_utils import length_hint, peek


class ArrayStream(Iterator):
    """
    Iterator of arrays. Elements from the stream are converted to
    NumPy arrays. If ``stream`` is a single array, it will be
    repackaged as a length 1 iterable.

    Arrays in the stream will be cast to the same data-type as the first
    array in the stream. The stream data-type is located in the `dtype` attribute.

    .. versionadded:: 1.5.2
    """

    def __init__(self, stream):
        if isinstance(stream, np.ndarray):
            stream = (stream,)

        self._sequence_length = length_hint(stream, default=NotImplemented)

        # Once length_hint has been determined, we can peek into the stream
        first, stream = peek(stream)
        self._iterator = iter(stream)

        first = asanyarray(first)
        self.dtype = first.dtype

    def __repr__(self):
        """Verbose string representation"""
        representation = f"< {self.__class__.__name__} object"
        representation += f" of data-type {self.dtype}"

        if not (self._sequence_length is NotImplemented):
            representation += f" and a sequence length of {self._sequence_length}"
        else:
            representation += " of unknown length"

        return representation + " >"

    def __array__(self, *_, **__):
        """Returns a dense array created from this stream."""
        # As of numpy version 1.14, arrays are expanded into a list before contatenation
        # Therefore, it's ok to build that list first
        arraylist = list(self)
        return np.stack(arraylist, axis=-1)

    def __length_hint__(self):
        """
        In certain cases, an ArrayStream can have a definite size.
        See https://www.python.org/dev/peps/pep-0424/
        """
        return self._sequence_length

    def __next__(self):
        n = self._iterator.__next__()
        return asanyarray(n, dtype=self.dtype)


def array_stream(func):
    """
    Decorates streaming functions to make sure that the stream
    is a stream of ndarrays. Objects that are not arrays are transformed
    into arrays. If the stream is in fact a single ndarray, this ndarray
    is repackaged into a sequence of length 1.

    The first argument of the decorated function is assumed to be an iterable of
    arrays, or an iterable of objects that can be casted to arrays.

    Note that using this decorator also ensures that the stream is only wrapped once
    by the conversion function.
    """

    @wraps(func)
    def decorated(arrays, *args, **kwargs):
        if isinstance(arrays, ArrayStream):
            return func(arrays, *args, **kwargs)
        return func(ArrayStream(arrays), *args, **kwargs)

    return decorated
