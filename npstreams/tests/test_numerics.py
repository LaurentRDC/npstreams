# -*- coding: utf-8 -*-

from random import randint, random

import numpy as np

from npstreams import isum, iprod, last, isub, iany, iall, prod
from npstreams import sum as nssum  # avoiding name clashes
import pytest


def test_isum_trivial():
    """ Test a sum of zeros """
    source = [np.zeros((16,), dtype=np.float) for _ in range(10)]
    summed = last(isum(source))
    assert np.allclose(summed, np.zeros_like(summed))


def test_isum_ignore_nans():
    """ Test a sum of zeros with NaNs sprinkled """
    source = [np.zeros((16,), dtype=np.float) for _ in range(10)]
    source.append(np.full((16,), fill_value=np.nan))
    summed = last(isum(source, ignore_nan=True))
    assert np.allclose(summed, np.zeros_like(summed))


def test_isum_length():
    """ Test that the number of yielded elements is the same as source """
    source = [np.zeros((16,), dtype=np.float) for _ in range(10)]
    summed = list(isum(source, axis=0))
    assert 10 == len(summed)


def test_isum_dtype():
    """ Test a sum of floating zeros with an int accumulator """
    source = [np.zeros((16,), dtype=np.float) for _ in range(10)]
    summed = last(isum(source, dtype=np.int))
    assert np.allclose(summed, np.zeros_like(summed))
    assert summed.dtype == np.int


def test_isum_axis():
    """ Test that isum(axis = 0) yields 0d arrays """
    source = [np.zeros((16,), dtype=np.float) for _ in range(10)]

    summed = last(isum(source, axis=0))
    assert np.allclose(summed, np.zeros_like(summed))

    summed = last(isum(source, axis=None))
    assert np.allclose(summed, 0)


def test_isum_return_shape():
    """ Test that the shape of output is as expected """
    source = [np.zeros((16,), dtype=np.float) for _ in range(10)]

    summed = last(isum(source, axis=0))
    assert summed.shape == (1, 10)


@pytest.mark.parametrize("axis", (0, 1, 2, None))
def test_isum_against_numpy(axis):
    """ Test that isum() returns the same as numpy.sum() for various axis inputs """

    stream = [np.random.random((16, 16)) for _ in range(10)]
    stack = np.dstack(stream)

    from_numpy = np.sum(stack, axis=axis)
    from_isum = last(isum(stream, axis=axis))
    assert np.allclose(from_isum, from_numpy)


def test_sum_trivial():
    """ Test a sum of zeros """
    source = [np.zeros((16,), dtype=np.float) for _ in range(10)]
    summed = nssum(source)
    assert np.allclose(summed, np.zeros_like(summed))


def test_sum_ignore_nans():
    """ Test a sum of zeros with NaNs sprinkled """
    source = [np.zeros((16,), dtype=np.float) for _ in range(10)]
    source.append(np.full((16,), fill_value=np.nan))
    summed = nssum(source, ignore_nan=True)
    assert np.allclose(summed, np.zeros_like(summed))


def test_sum_dtype():
    """ Test a sum of floating zeros with an int accumulator """
    source = [np.zeros((16,), dtype=np.float) for _ in range(10)]
    summed = nssum(source, dtype=np.int)
    assert np.allclose(summed, np.zeros_like(summed))
    assert summed.dtype == np.int


def test_sum_axis():
    """ Test that isum(axis = 0) yields 0d arrays """
    source = [np.zeros((16,), dtype=np.float) for _ in range(10)]

    summed = nssum(source, axis=0)
    assert np.allclose(summed, np.zeros_like(summed))

    summed = nssum(source, axis=None)
    assert np.allclose(summed, 0)


def test_sum_return_shape():
    """ Test that the shape of output is as expected """
    source = [np.zeros((16,), dtype=np.float) for _ in range(10)]

    summed = nssum(source, axis=0)
    assert summed.shape == (1, 10)


@pytest.mark.parametrize("axis", (0, 1, 2, None))
def test_sum_against_numpy(axis):
    """ Test that isum() returns the same as numpy.sum() for various axis inputs """

    stream = [np.random.random((16, 16)) for _ in range(10)]
    stack = np.dstack(stream)

    from_numpy = np.sum(stack, axis=axis)
    from_sum = nssum(stream, axis=axis)
    assert np.allclose(from_sum, from_numpy)


def test_iprod_trivial():
    """ Test a product of ones """
    source = [np.ones((16,), dtype=np.float) for _ in range(10)]
    product = last(iprod(source))
    assert np.allclose(product, np.ones_like(product))


def test_iprod_ignore_nans():
    """ Test that NaNs are ignored. """
    source = [np.ones((16,), dtype=np.float) for _ in range(10)]
    source.append(np.full_like(source[0], np.nan))
    product = last(iprod(source, ignore_nan=True))
    assert np.allclose(product, np.ones_like(product))


def test_iprod_dtype():
    """ Test that dtype argument is working """
    source = [np.ones((16,), dtype=np.float) for _ in range(10)]
    product = last(iprod(source, dtype=np.int))
    assert np.allclose(product, np.ones_like(product))
    assert product.dtype == np.int


def test_iprod_axis():
    """ Test that iprod(axis = 0) yields 0d arrays """
    source = [np.ones((16,), dtype=np.float) for _ in range(10)]

    summed = last(iprod(source, axis=0))
    assert np.all(summed == 1)

    summed = last(iprod(source, axis=None))
    assert np.allclose(summed, np.ones_like(summed))


@pytest.mark.parametrize("axis", (0, 1, 2, None))
def test_iprod_against_numpy(axis):
    """ Test that iprod() returns the same as numpy.prod() for various axis inputs """

    stream = [np.random.random((16, 16)) for _ in range(10)]
    stack = np.dstack(stream)

    from_numpy = np.prod(stack, axis=axis)
    from_stream = last(iprod(stream, axis=axis))
    assert np.allclose(from_stream, from_numpy)


def test_prod_trivial():
    """ Test a product of ones """
    source = [np.ones((16,), dtype=np.float) for _ in range(10)]
    product = prod(source)
    assert np.allclose(product, np.ones_like(product))


def test_prod_ignore_nans():
    """ Test that NaNs are ignored. """
    source = [np.ones((16,), dtype=np.float) for _ in range(10)]
    source.append(np.full_like(source[0], np.nan))
    product = prod(source, ignore_nan=True)
    assert np.allclose(product, np.ones_like(product))


def test_prod_dtype():
    """ Test that dtype argument is working """
    source = [np.ones((16,), dtype=np.float) for _ in range(10)]
    product = prod(source, dtype=np.int)
    assert np.allclose(product, np.ones_like(product))
    assert product.dtype == np.int


def test_prod_axis():
    """ Test that iprod(axis = 0) yields 0d arrays """
    source = [np.ones((16,), dtype=np.float) for _ in range(10)]

    summed = prod(source, axis=0)
    assert np.all(summed == 1)

    summed = prod(source, axis=None)
    assert np.allclose(summed, np.ones_like(summed))


@pytest.mark.parametrize("axis", (0, 1, 2, None))
def test_prod_against_numpy(axis):
    """ Test that iprod() returns the same as numpy.prod() for various axis inputs """

    stream = [np.random.random((16, 16)) for _ in range(10)]
    stack = np.dstack(stream)

    from_numpy = np.prod(stack, axis=axis)
    from_stream = prod(stream, axis=axis)
    assert np.allclose(from_stream, from_numpy)


@pytest.mark.parametrize("axis", (0, 1, 2))
def test_isub_against_numpy(axis):
    """ Test against numpy.subtract.reduce """
    stream = [np.random.random((8, 16, 2)) for _ in range(11)]
    stack = np.stack(stream, axis=-1)

    from_numpy = np.subtract.reduce(stack, axis=axis)
    from_stream = last(isub(stream, axis=axis))
    assert np.allclose(from_numpy, from_stream)


@pytest.mark.parametrize("axis", (0, 1, 2, None))
def test_iall_against_numpy(axis):
    """ Test iall against numpy.all """
    stream = [np.zeros((8, 16, 2)) for _ in range(11)]
    stream[3][3, 0, 1] = 1  # so that np.all(axis = None) evaluates to False
    stack = np.stack(stream, axis=-1)

    from_numpy = np.all(stack, axis=axis)
    from_stream = last(iall(stream, axis=axis))
    assert np.allclose(from_numpy, from_stream)


@pytest.mark.parametrize("axis", (0, 1, 2, None))
def test_iany_against_numpy(axis):
    """ Test iany against numpy.any """
    stream = [np.zeros((8, 16, 2)) for _ in range(11)]
    stream[3][3, 0, 1] = 1  # so that np.all(axis = None) evaluates to False
    stack = np.stack(stream, axis=-1)

    from_numpy = np.any(stack, axis=axis)
    from_stream = last(iany(stream, axis=axis))
    assert np.allclose(from_numpy, from_stream)
