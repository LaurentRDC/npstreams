# -*- coding: utf-8 -*-

from itertools import repeat
from random import randint, random, seed
from warnings import catch_warnings, simplefilter
import pytest

import numpy as np

try:
    from scipy.stats import sem as scipy_sem

    WITH_SCIPY = True
except ImportError:
    WITH_SCIPY = False

from npstreams import (
    iaverage,
    imean,
    isem,
    istd,
    ivar,
    last,
    ihistogram,
    mean,
    average,
    sem,
    std,
    var,
)

seed(23)


def test_average_trivial():
    """ Test average() on a stream of zeroes """
    stream = repeat(np.zeros((64, 64), dtype=float), times=5)
    for av in average(stream):
        assert np.allclose(av, np.zeros_like(av))


@pytest.mark.parametrize("axis", (0, 1, 2, None))
def test_average_vs_numpy(axis):
    """ Test average vs. numpy.average """
    stream = [np.random.random(size=(64, 64)) for _ in range(5)]
    stack = np.dstack(stream)

    from_stream = average(stream, axis=axis)
    from_numpy = np.average(stack, axis=axis)
    assert np.allclose(from_numpy, from_stream)


def test_average_weighted_average():
    """ Test results of weighted average against numpy.average """
    stream = [np.random.random(size=(16, 16)) for _ in range(5)]

    weights = [random() for _ in stream]
    from_average = average(stream, weights=weights)
    from_numpy = np.average(np.dstack(stream), axis=2, weights=np.array(weights))
    assert np.allclose(from_average, from_numpy)

    weights = [np.random.random(size=stream[0].shape) for _ in stream]
    from_average = average(stream, weights=weights)
    from_numpy = np.average(np.dstack(stream), axis=2, weights=np.dstack(weights))
    assert np.allclose(from_average, from_numpy)


def test_average_ignore_nan():
    """ Test that NaNs are handled correctly """
    stream = [np.random.random(size=(16, 12)) for _ in range(5)]
    for s in stream:
        s[randint(0, 15), randint(0, 11)] = np.nan

    with catch_warnings():
        simplefilter("ignore")
        from_average = average(stream, ignore_nan=True)
    from_numpy = np.nanmean(np.dstack(stream), axis=2)
    assert np.allclose(from_average, from_numpy)


def test_iaverage_trivial():
    """ Test iaverage on stream of zeroes """
    stream = repeat(np.zeros((64, 64), dtype=float), times=5)
    for av in iaverage(stream):
        assert np.allclose(av, np.zeros_like(av))


def test_iaverage_weighted_average():
    """ Test results of weighted iverage against numpy.average """
    stream = [np.random.random(size=(16, 16)) for _ in range(5)]

    weights = [random() for _ in stream]
    from_iaverage = last(iaverage(stream, weights=weights))
    from_numpy = np.average(np.dstack(stream), axis=2, weights=np.array(weights))
    assert np.allclose(from_iaverage, from_numpy)

    weights = [np.random.random(size=stream[0].shape) for _ in stream]
    from_iaverage = last(iaverage(stream, weights=weights))
    from_numpy = np.average(np.dstack(stream), axis=2, weights=np.dstack(weights))
    assert np.allclose(from_iaverage, from_numpy)


def test_iaverage_ignore_nan():
    """ Test that NaNs are handled correctly """
    stream = [np.random.random(size=(16, 12)) for _ in range(5)]
    for s in stream:
        s[randint(0, 15), randint(0, 11)] = np.nan

    with catch_warnings():
        simplefilter("ignore")
        from_iaverage = last(iaverage(stream, ignore_nan=True))
    from_numpy = np.nanmean(np.dstack(stream), axis=2)
    assert np.allclose(from_iaverage, from_numpy)


def test_iaverage_length():
    """ Test that the number of yielded elements is the same as source """
    source = (np.zeros((16,)) for _ in range(5))
    avg = list(iaverage(source, axis=0))
    assert len(avg) == 5


@pytest.mark.parametrize("dtype", (np.uint8, bool, np.int16, np.float16))
def test_iaverage_output_dtype(dtype):
    """ Test that that yielded arrays are always floats """
    source = (np.zeros((16,), dtype=dtype) for _ in range(5))
    avg = last(iaverage(source))
    assert avg.dtype == float


@pytest.mark.parametrize("axis", (0, 1, 2, None))
def test_iaverage_output_shape(axis):
    """ Test output shape """
    source = [np.random.random((16, 12, 5)) for _ in range(10)]
    stack = np.stack(source, axis=-1)

    from_numpy = np.average(stack, axis=axis)
    out = last(iaverage(source, axis=axis))
    assert from_numpy.shape == out.shape
    assert np.allclose(out, from_numpy)


def test_mean_trivial():
    """ Test mean() on a stream of zeroes """
    stream = repeat(np.zeros((64, 64), dtype=float), times=5)
    for av in mean(stream):
        assert np.allclose(av, np.zeros_like(av))


@pytest.mark.parametrize("axis", (0, 1, 2, None))
def test_mean_vs_numpy(axis):
    """ Test mean vs. numpy.mean """
    stream = [np.random.random(size=(64, 64)) for _ in range(5)]
    stack = np.dstack(stream)

    from_stream = mean(stream, axis=axis)
    from_numpy = np.mean(stack, axis=axis)
    assert np.allclose(from_numpy, from_stream)


@pytest.mark.parametrize("axis", (0, 1, 2, None))
def test_mean_against_numpy_nanmean(axis):
    """ Test results against numpy.mean"""
    source = [np.random.random((16, 12, 5)) for _ in range(10)]
    for arr in source:
        arr[randint(0, 15), randint(0, 11), randint(0, 4)] = np.nan
    stack = np.stack(source, axis=-1)

    from_numpy = np.nanmean(stack, axis=axis)
    out = mean(source, axis=axis, ignore_nan=True)
    assert from_numpy.shape == out.shape
    assert np.allclose(out, from_numpy)


@pytest.mark.parametrize("axis", (0, 1, 2, None))
def test_imean_against_numpy_mean(axis):
    """ Test results against numpy.mean"""
    source = [np.random.random((16, 12, 5)) for _ in range(10)]
    stack = np.stack(source, axis=-1)

    from_numpy = np.mean(stack, axis=axis)
    out = last(imean(source, axis=axis))
    assert from_numpy.shape == out.shape
    assert np.allclose(out, from_numpy)


@pytest.mark.parametrize("axis", (0, 1, 2, None))
def test_imean_against_numpy_nanmean(axis):
    """ Test results against numpy.mean"""
    source = [np.random.random((16, 12, 5)) for _ in range(10)]
    for arr in source:
        arr[randint(0, 15), randint(0, 11), randint(0, 4)] = np.nan
    stack = np.stack(source, axis=-1)

    from_numpy = np.nanmean(stack, axis=axis)
    out = last(imean(source, axis=axis, ignore_nan=True))
    assert from_numpy.shape == out.shape
    assert np.allclose(out, from_numpy)


@pytest.mark.parametrize("axis", (0, 1, 2, None))
def test_var_vs_numpy(axis):
    """ Test that the axis parameter is handled correctly """
    stream = [np.random.random((16, 7, 3)) for _ in range(5)]
    stack = np.stack(stream, axis=-1)

    from_numpy = np.var(stack, axis=axis)
    from_var = var(stream, axis=axis)
    assert from_numpy.shape == from_var.shape
    assert np.allclose(from_var, from_numpy)


@pytest.mark.parametrize("axis", (0, 1, 2, None))
@pytest.mark.parametrize("ddof", range(4))
def test_var_ddof(axis, ddof):
    """ Test that the ddof parameter is equivalent to numpy's """
    stream = [np.random.random((16, 7, 3)) for _ in range(10)]
    stack = np.stack(stream, axis=-1)

    with catch_warnings():
        simplefilter("ignore")

        from_numpy = np.var(stack, axis=axis, ddof=ddof)
        from_var = var(stream, axis=axis, ddof=ddof)
        assert from_numpy.shape == from_var.shape
        assert np.allclose(from_var, from_numpy)


def test_ivar_first():
    """ Test that the first yielded value of ivar is an array fo zeros """
    stream = repeat(np.random.random(size=(64, 64)), times=5)
    first = next(ivar(stream))

    assert np.allclose(first, np.zeros_like(first))


@pytest.mark.parametrize("axis", (0, 1, 2, None))
def test_ivar_output_shape(axis):
    """ Test that the axis parameter is handled correctly """
    stream = [np.random.random((16, 7, 3)) for _ in range(5)]
    stack = np.stack(stream, axis=-1)

    from_numpy = np.var(stack, axis=axis)
    from_ivar = last(ivar(stream, axis=axis))
    assert from_numpy.shape == from_ivar.shape
    assert np.allclose(from_ivar, from_numpy)


@pytest.mark.parametrize("axis", (0, 1, 2, None))
@pytest.mark.parametrize("ddof", range(4))
def test_ivar_ddof(axis, ddof):
    """ Test that the ddof parameter is equivalent to numpy's """
    stream = [np.random.random((16, 7, 3)) for _ in range(10)]
    stack = np.stack(stream, axis=-1)

    with catch_warnings():
        simplefilter("ignore")

        from_numpy = np.var(stack, axis=axis, ddof=ddof)
        from_ivar = last(ivar(stream, axis=axis, ddof=ddof))
        assert from_numpy.shape == from_ivar.shape
        assert np.allclose(from_ivar, from_numpy)


@pytest.mark.parametrize("axis", (0, 1, 2, None))
@pytest.mark.parametrize("ddof", range(4))
def test_std_against_numpy_std(axis, ddof):
    stream = [np.random.random((16, 7, 3)) for _ in range(10)]
    stack = np.stack(stream, axis=-1)

    with catch_warnings():
        simplefilter("ignore")

        from_numpy = np.std(stack, axis=axis, ddof=ddof)
        from_ivar = std(stream, axis=axis, ddof=ddof)
        assert from_numpy.shape == from_ivar.shape
        assert np.allclose(from_ivar, from_numpy)


@pytest.mark.parametrize("axis", (0, 1, 2, None))
@pytest.mark.parametrize("ddof", range(4))
def test_std_against_numpy_nanstd(axis, ddof):
    source = [np.random.random((16, 12, 5)) for _ in range(10)]
    for arr in source:
        arr[randint(0, 15), randint(0, 11), randint(0, 4)] = np.nan
    stack = np.stack(source, axis=-1)

    from_numpy = np.nanstd(stack, axis=axis, ddof=ddof)
    from_ivar = std(source, axis=axis, ddof=ddof, ignore_nan=True)
    assert from_numpy.shape == from_ivar.shape
    assert np.allclose(from_ivar, from_numpy)


@pytest.mark.parametrize("axis", (0, 1, 2, None))
@pytest.mark.parametrize("ddof", range(4))
def test_istd_against_numpy_std(axis, ddof):
    stream = [np.random.random((16, 7, 3)) for _ in range(10)]
    stack = np.stack(stream, axis=-1)

    with catch_warnings():
        simplefilter("ignore")

        from_numpy = np.std(stack, axis=axis, ddof=ddof)
        from_ivar = last(istd(stream, axis=axis, ddof=ddof))
        assert from_numpy.shape == from_ivar.shape
        assert np.allclose(from_ivar, from_numpy)


@pytest.mark.parametrize("axis", (0, 1, 2, None))
@pytest.mark.parametrize("ddof", range(4))
def test_istd_against_numpy_nanstd(axis, ddof):
    source = [np.random.random((16, 12, 5)) for _ in range(10)]
    for arr in source:
        arr[randint(0, 15), randint(0, 11), randint(0, 4)] = np.nan
    stack = np.stack(source, axis=-1)

    from_numpy = np.nanstd(stack, axis=axis, ddof=ddof)
    from_ivar = last(istd(source, axis=axis, ddof=ddof, ignore_nan=True))
    assert from_numpy.shape == from_ivar.shape
    assert np.allclose(from_ivar, from_numpy)


@pytest.mark.skipif(not WITH_SCIPY, reason="SciPy is not installed/importable")
@pytest.mark.parametrize("axis", (0, 1, 2, None))
@pytest.mark.parametrize("ddof", range(4))
def test_sem_against_scipy_no_nans(axis, ddof):
    """ Test that isem outputs the same as scipy.stats.sem """
    source = [np.random.random((16, 12, 5)) for _ in range(10)]
    stack = np.stack(source, axis=-1)

    from_scipy = scipy_sem(stack, axis=axis, ddof=ddof)
    from_isem = sem(source, axis=axis, ddof=ddof)
    assert from_scipy.shape == from_isem.shape
    assert np.allclose(from_isem, from_scipy)


@pytest.mark.skipif(not WITH_SCIPY, reason="SciPy is not installed/importable")
@pytest.mark.parametrize("axis", (0, 1, 2, None))
@pytest.mark.parametrize("ddof", range(4))
def test_sem_against_scipy_with_nans(axis, ddof):
    """ Test that isem outputs the same as scipy.stats.sem when NaNs are ignored. """
    source = [np.random.random((16, 12, 5)) for _ in range(10)]
    for arr in source:
        arr[randint(0, 15), randint(0, 11), randint(0, 4)] = np.nan
    stack = np.stack(source, axis=-1)

    from_scipy = scipy_sem(stack, axis=axis, ddof=ddof, nan_policy="omit")
    from_isem = sem(source, axis=axis, ddof=ddof, ignore_nan=True)
    assert from_scipy.shape == from_isem.shape
    assert np.allclose(from_isem, from_scipy)


@pytest.mark.skipif(not WITH_SCIPY, reason="SciPy is not installed/importable")
@pytest.mark.parametrize("axis", (0, 1, 2, None))
@pytest.mark.parametrize("ddof", range(4))
def test_isem_against_scipy_no_nans(axis, ddof):
    """ Test that isem outputs the same as scipy.stats.sem """
    source = [np.random.random((16, 12, 5)) for _ in range(10)]
    stack = np.stack(source, axis=-1)

    from_scipy = scipy_sem(stack, axis=axis, ddof=ddof)
    from_isem = last(isem(source, axis=axis, ddof=ddof))
    assert from_scipy.shape == from_isem.shape
    assert np.allclose(from_isem, from_scipy)


@pytest.mark.skipif(not WITH_SCIPY, reason="SciPy is not installed/importable")
@pytest.mark.parametrize("axis", (0, 1, 2, None))
@pytest.mark.parametrize("ddof", range(4))
def test_isem_against_scipy_with_nans(axis, ddof):
    """ Test that isem outputs the same as scipy.stats.sem when NaNs are ignored. """
    source = [np.random.random((16, 12, 5)) for _ in range(10)]
    for arr in source:
        arr[randint(0, 15), randint(0, 11), randint(0, 4)] = np.nan
    stack = np.stack(source, axis=-1)

    from_scipy = scipy_sem(stack, axis=axis, ddof=ddof, nan_policy="omit")
    from_isem = last(isem(source, axis=axis, ddof=ddof, ignore_nan=True))
    assert from_scipy.shape == from_isem.shape
    assert np.allclose(from_isem, from_scipy)


def test_ihistogram_against_numpy_no_weights():
    """ Test ihistogram against numpy.histogram with no weights """
    source = [np.random.random((16, 12, 5)) for _ in range(10)]
    stack = np.stack(source, axis=-1)

    bins = np.linspace(0, 1, num=10)
    from_numpy = np.histogram(stack, bins=bins)[0]
    from_ihistogram = last(ihistogram(source, bins=bins))

    # Since histogram output is int, cannot use allclose
    assert np.all(np.equal(from_numpy, from_ihistogram))


def test_ihistogram_trivial_weights():
    """ Test ihistogram with weights being all 1s vs. weights=None """
    source = [np.random.random((16, 12, 5)) for _ in range(10)]
    weights = [np.array([1]) for _ in source]

    bins = np.linspace(0, 1, num=10)
    none_weights = last(ihistogram(source, bins=bins, weights=None))
    trivial_weights = last(ihistogram(source, bins=bins, weights=weights))

    assert np.all(np.equal(none_weights, trivial_weights))
