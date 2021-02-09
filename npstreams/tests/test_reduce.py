# -*- coding: utf-8 -*-

import numpy as np

from npstreams import ireduce_ufunc, preduce_ufunc, last, nan_to_num, reduce_ufunc
import pytest

# Only testing binary ufuncs that support floats
# i.e. leaving bitwise_* and logical_* behind
# Also, numpy.ldexp takes in ints and floats separately, so
# leave it behind
UFUNCS = (
    np.add,
    np.subtract,
    np.multiply,
    np.divide,
    np.logaddexp,
    np.logaddexp2,
    np.true_divide,
    np.floor_divide,
    np.power,
    np.remainder,
    np.mod,
    np.fmod,
    np.arctan2,
    np.hypot,
    np.maximum,
    np.fmax,
    np.minimum,
    np.fmin,
    np.copysign,
    np.nextafter,
)

UFUNCS_WITH_IDENTITY = list(filter(lambda u: u.identity is not None, UFUNCS))


def test_ireduce_ufunc_no_side_effects():
    """ Test that no arrays in the stream are modified """
    source = [np.random.random((16, 5, 8)) for _ in range(10)]
    stack = np.stack(source, axis=-1)
    for arr in source:
        arr.setflags(write=False)
    out = last(ireduce_ufunc(source, np.add))


def test_ireduce_ufunc_single_array():
    """ Test ireduce_ufunc on a single array, not a sequence """
    source = [np.random.random((16, 5, 8)) for _ in range(10)]
    stack = np.stack(source, axis=-1)
    source = np.ones((16, 16), dtype=int)
    out = last(ireduce_ufunc(source, np.add, axis=-1))
    assert np.allclose(source, out)


def test_ireduce_ufunc_out_parameter():
    """ Test that the kwargs ``out`` is correctly passed to reduction function """
    source = [np.random.random((16, 5, 8)) for _ in range(10)]
    stack = np.stack(source, axis=-1)
    not_out = last(ireduce_ufunc(source, np.add, axis=-1))
    out = np.empty_like(source[0])
    last(ireduce_ufunc(source, ufunc=np.add, out=out))

    assert np.allclose(not_out, out)

    not_out = last(ireduce_ufunc(source, np.add, axis=2))
    out = np.empty_like(source[0])
    from_out = last(ireduce_ufunc(source, ufunc=np.add, out=out, axis=2))

    assert np.allclose(not_out, from_out)


def test_ireduce_ufunc_ignore_nan_no_identity():
    """Test ireduce_ufunc on an ufunc with no identity raises
    an error for ignore_nan = True"""
    source = [np.ones((16, 16), dtype=int) for _ in range(5)]
    with pytest.raises(ValueError):
        ireduce_ufunc(source, np.maximum, axis=-1, ignore_nan=True)


def test_ireduce_ufunc_non_ufunc():
    """ Test that ireduce_ufunc raises TypeError when a non-ufunc is passed """
    with pytest.raises(TypeError):
        ireduce_ufunc(range(10), ufunc=lambda x: x)


def test_ireduce_ufunc_non_binary_ufunc():
    """ Test that ireduce_ufunc raises ValueError if non-binary ufunc is used """
    with pytest.raises(ValueError):
        ireduce_ufunc(range(10), ufunc=np.absolute)


@pytest.mark.parametrize("axis", (0, 1, 2, 3, None))
def test_ireduce_ufunc_output_shape(axis):
    """ Test output shape """
    source = [np.random.random((16, 5, 8)) for _ in range(10)]
    stack = np.stack(source, axis=-1)

    from_numpy = np.add.reduce(stack, axis=axis)
    out = last(ireduce_ufunc(source, np.add, axis=axis))
    assert from_numpy.shape == out.shape
    assert np.allclose(out, from_numpy)


@pytest.mark.parametrize("axis", (0, 1, 2, 3, None))
def test_ireduce_ufunc_length(axis):
    """ Test that the number of elements yielded by ireduce_ufunc is correct """

    source = (np.zeros((16, 5, 8)) for _ in range(10))
    out = list(ireduce_ufunc(source, np.add, axis=axis))
    assert 10 == len(out)


@pytest.mark.parametrize("axis", (0, 1, 2, 3, None))
def test_ireduce_ufunc_ignore_nan(axis):
    """ Test that ignore_nan is working """
    source = [np.random.random((16, 5, 8)) for _ in range(10)]
    stack = np.stack(source, axis=-1)

    out = last(ireduce_ufunc(source, np.add, axis=axis, ignore_nan=True))
    assert not np.any(np.isnan(out))


def test_preduce_ufunc_trivial():
    """ Test preduce_ufunc for a sum of zeroes over two processes"""
    stream = [np.zeros((8, 8)) for _ in range(10)]
    s = preduce_ufunc(stream, ufunc=np.add, processes=2, ntotal=10)
    assert np.allclose(s, np.zeros_like(s))


def test_preduce_ufunc_correctess():
    """ Test preduce_ufunc is equivalent to reduce_ufunc for random sums"""
    stream = [np.random.random((8, 8)) for _ in range(20)]
    s = preduce_ufunc(stream, ufunc=np.add, processes=3, ntotal=10)
    assert np.allclose(s, reduce_ufunc(stream, np.add))


# Dynamics generation of tests on binary ufuncs
@pytest.mark.parametrize("ufunc", UFUNCS)
@pytest.mark.parametrize("axis", (0, 1, 2, -1))
def test_binary_ufunc(ufunc, axis):
    """Generate a test to ensure that ireduce_ufunc(..., ufunc, ...)
    works as intendent."""
    source = [np.random.random((16, 5, 8)) for _ in range(10)]
    stack = np.stack(source, axis=-1)

    def sufunc(arrays, axis=-1):  # s for stream
        return last(ireduce_ufunc(arrays, ufunc, axis=axis))

    from_numpy = ufunc.reduce(stack, axis=axis)
    from_sufunc = sufunc(source, axis=axis)
    assert from_sufunc.shape == from_numpy.shape
    assert np.allclose(from_numpy, from_sufunc)


@pytest.mark.parametrize("ufunc", UFUNCS_WITH_IDENTITY)
def test_binary_ufunc_ignore_nan(ufunc):
    """Generate a test to ensure that ireduce_ufunc(..., ufunc, ...)
    works as intendent with NaNs in stream."""

    source = [np.random.random((16, 5, 8)) for _ in range(10)]
    source[0][0, 0, 0] = np.nan
    stack = nan_to_num(np.stack(source, axis=-1), fill_value=ufunc.identity)

    def sufunc(arrays, ignore_nan=False):  # s for stream
        return last(ireduce_ufunc(arrays, ufunc, axis=1, ignore_nan=True))

    from_numpy = ufunc.reduce(stack, axis=1)
    from_sufunc = sufunc(source)
    assert from_numpy.shape == from_sufunc.shape
    assert np.allclose(from_numpy, from_sufunc)
