# -*- coding: utf-8 -*-
from npstreams import pmap, pmap_unordered, preduce
from functools import reduce
import numpy as np
from operator import add


def identity(obj, *args, **kwargs):
    """ignores args and kwargs"""
    return obj


def test_preduce_preduce_one_process():
    """Test that preduce reduces to functools.reduce for a single process"""
    integers = list(range(0, 10))
    preduce_results = preduce(add, integers, processes=1)
    reduce_results = reduce(add, integers)

    assert preduce_results == reduce_results


def test_preduce_preduce_multiple_processes():
    """Test that preduce reduces to functools.reduce for a single process"""
    integers = list(range(0, 10))
    preduce_results = preduce(add, integers, processes=2)
    reduce_results = reduce(add, integers)

    assert preduce_results == reduce_results


def test_preduce_on_numpy_arrays():
    """Test sum of numpy arrays as parallel reduce"""
    arrays = [np.zeros((32, 32)) for _ in range(10)]
    s = preduce(add, arrays, processes=2)

    assert np.allclose(s, arrays[0])


def test_preduce_with_kwargs():
    """Test preduce with keyword-arguments"""
    pass


def test_pmap_trivial_map_no_args():
    """Test that pmap is working with no positional arguments"""
    integers = list(range(0, 10))
    result = list(pmap(identity, integers, processes=2))
    assert integers == result


def test_pmap_trivial_map_kwargs():
    """Test that pmap is working with args and kwargs"""
    integers = list(range(0, 10))
    result = list(pmap(identity, integers, processes=2, kwargs={"test": True}))
    assert result == integers


def test_pmap_trivial_map_no_args():
    """Test that pmap_unordered is working with no positional arguments"""
    integers = list(range(0, 10))
    result = list(sorted(pmap_unordered(identity, integers, processes=2)))
    assert integers == result


def test_pmap_trivial_map_kwargs():
    """Test that pmap_unordered is working with args and kwargs"""
    integers = list(range(0, 10))
    result = list(
        sorted(pmap_unordered(identity, integers, processes=2, kwargs={"test": True}))
    )
    assert result == integers
