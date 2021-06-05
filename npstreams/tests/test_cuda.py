# -*- coding: utf-8 -*-

from itertools import repeat
import numpy as np
import pytest

try:
    from npstreams.cuda import csum, cprod, caverage, cmean

    WITH_CUDA = True
except ImportError:
    WITH_CUDA = False


skip_if_no_cuda = pytest.mark.skipif(
    not WITH_CUDA, reason="PyCUDA is not installed/available"
)


@skip_if_no_cuda
def test_csum_zero_sum():
    stream = repeat(np.zeros((16, 16), dtype=float), times=5)
    s = csum(stream)
    assert np.allclose(s, np.zeros((16, 16)))


@skip_if_no_cuda
def test_csum_dtype():
    stream = repeat(np.zeros((16, 16), dtype=float), times=5)
    s = csum(stream, dtype=np.int16)
    assert np.allclose(s, np.zeros((16, 16)))
    assert s.dtype == np.int16


@skip_if_no_cuda
def test_csum_ignore_nans():
    """Test a sum of zeros with NaNs sprinkled"""
    source = [np.zeros((16,), dtype=float) for _ in range(10)]
    source.append(np.full((16,), fill_value=np.nan))
    summed = csum(source, ignore_nan=True)
    assert np.allclose(summed, np.zeros_like(summed))


@skip_if_no_cuda
def test_cprod_ones_prod():
    stream = repeat(np.ones((16, 16), dtype=float), times=5)
    s = cprod(stream)
    assert np.allclose(s, np.ones((16, 16)))


@skip_if_no_cuda
def test_cprod_ignore_nans():
    """Test that NaNs are ignored."""
    source = [np.ones((16,), dtype=float) for _ in range(10)]
    source.append(np.full_like(source[0], np.nan))
    product = cprod(source, ignore_nan=True)
    assert np.allclose(product, np.ones_like(product))


@skip_if_no_cuda
def test_cprod_dtype():
    """Test that dtype argument is working"""
    source = [np.ones((16,), dtype=float) for _ in range(10)]
    product = cprod(source, dtype=int)
    assert np.allclose(product, np.ones_like(product))
    assert product.dtype == int


@skip_if_no_cuda
def test_cavg_no_weights():
    stream = [np.random.random(size=(16, 16)) for _ in range(5)]
    from_caverage = caverage(stream)
    from_numpy = np.average(np.dstack(stream), axis=2)
    assert np.allclose(from_caverage, from_numpy)


@skip_if_no_cuda
def test_cavg_weighted_average():
    """Test results of weighted average against numpy.average"""
    stream = [np.random.random(size=(16, 16)) for _ in range(5)]

    weights = [np.random.random(size=stream[0].shape) for _ in stream]
    from_caverage = caverage(stream, weights=weights)
    from_numpy = np.average(np.dstack(stream), axis=2, weights=np.dstack(weights))
    assert np.allclose(from_caverage, from_numpy)


@skip_if_no_cuda
def test_cmean_of_ones():
    stream = repeat(np.ones((16, 16), dtype=float), times=5)
    s = cmean(stream)
    assert np.allclose(s, np.ones((16, 16)))


@skip_if_no_cuda
def test_cmean_random():
    """Test cmean against numpy.mean on random data"""
    stream = [np.random.random(size=(16, 16)) for _ in range(5)]
    from_cmean = cmean(stream)
    from_numpy = np.mean(np.dstack(stream), axis=2)
    assert np.allclose(from_cmean, from_numpy)
