# -*- coding: utf-8 -*-

import numpy as np

from npstreams import stack
import pytest


def test_stack_against_numpy_stack():
    """ Test against numpy.stack for axis = -1 and """
    stream = [np.random.random((15, 7, 2, 1)) for _ in range(10)]

    dense = np.stack(stream, axis=-1)
    from_stack = stack(stream, axis=-1)
    assert np.allclose(dense, from_stack)


def test_stack_on_single_array():
    """ Test that npstreams.stack works with a single array """
    arr = np.random.random((16, 16))
    stacked = stack(arr)
    assert np.allclose(arr[..., np.newaxis], stacked)


@pytest.mark.parametrize("axis", range(4))
def test_stack_against_numpy_concatenate(axis):
    """ Test against numpy.concatenate for existing axes """
    stream = [np.random.random((15, 7, 2, 1)) for _ in range(10)]

    dense = np.concatenate(stream, axis=axis)
    from_stack = stack(stream, axis=axis)
    assert np.allclose(dense, from_stack)
