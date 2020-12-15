# -*- coding: utf-8 -*-

from random import randint, random

import numpy as np

from npstreams import idot, itensordot, iinner, ieinsum, last
import pytest


def test_idot_against_numpy_multidot():
    """ Test against numpy.linalg.multi_dot in 2D case """
    stream = [np.random.random((8, 8)) for _ in range(7)]

    from_numpy = np.linalg.multi_dot(stream)
    from_stream = last(idot(stream))

    assert from_numpy.shape == from_stream.shape
    assert np.allclose(from_numpy, from_stream)


@pytest.mark.parametrize("axis", (0, 1, 2))
def test_itensordot_against_numpy_tensordot(axis):
    """ Test against numpy.tensordot in 2D case """
    stream = tuple(np.random.random((8, 8)) for _ in range(2))

    from_numpy = np.tensordot(*stream)
    from_stream = last(itensordot(stream))

    assert from_numpy.shape == from_stream.shape
    assert np.allclose(from_numpy, from_stream)


@pytest.mark.parametrize("axis", (0, 1, 2))
def test_iinner_against_numpy_inner(axis):
    """ Test against numpy.tensordot in 2D case """
    stream = tuple(np.random.random((8, 8)) for _ in range(2))

    from_numpy = np.inner(*stream)
    from_stream = last(iinner(stream))

    assert from_numpy.shape == from_stream.shape
    assert np.allclose(from_numpy, from_stream)


def test_ieinsum_against_numpy_einsum():
    """ Test against numpy.einsum  """
    a = np.arange(60.0).reshape(3, 4, 5)
    b = np.arange(24.0).reshape(4, 3, 2)
    stream = [a, b]

    from_numpy = np.einsum("ijk,jil->kl", a, b)
    from_stream = last(ieinsum(stream, "ijk,jil->kl"))

    assert from_numpy.shape == from_stream.shape
    assert np.allclose(from_numpy, from_stream)
