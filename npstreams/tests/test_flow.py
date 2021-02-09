# -*- coding: utf-8 -*-

import numpy as np
from pathlib import Path
from npstreams import array_stream, ipipe, last, iload, pload, isum


@array_stream
def iden(arrays):
    yield from arrays


def test_ipipe_order():
    """ Test that ipipe(f, g, h, arrays) -> f(g(h(arr))) for arr in arrays """
    stream = [np.random.random((15, 7, 2, 1)) for _ in range(10)]
    squared = [np.cbrt(np.square(arr)) for arr in stream]
    pipeline = ipipe(np.cbrt, np.square, stream)

    assert all(np.allclose(s, p) for s, p in zip(pipeline, squared))


def test_ipipe_multiprocessing():
    """ Test that ipipe(f, g, h, arrays) -> f(g(h(arr))) for arr in arrays """
    stream = [np.random.random((15, 7, 2, 1)) for _ in range(10)]
    squared = [np.cbrt(np.square(arr)) for arr in stream]
    pipeline = ipipe(np.cbrt, np.square, stream, processes=2)

    assert all(np.allclose(s, p) for s, p in zip(pipeline, squared))


def test_iload_glob():
    """ Test that iload works on glob-like patterns """
    stream = iload(Path(__file__).parent / "data" / "test_data*.npy", load_func=np.load)
    s = last(isum(stream)).astype(float)  # Cast to float for np.allclose
    assert np.allclose(s, np.zeros_like(s))


def test_iload_file_list():
    """ Test that iload works on iterable of filenames """
    files = [
        Path(__file__).parent / "data" / "test_data1.npy",
        Path(__file__).parent / "data" / "test_data2.npy",
        Path(__file__).parent / "data" / "test_data3.npy",
    ]
    stream = iload(files, load_func=np.load)
    s = last(isum(stream)).astype(float)  # Cast to float for np.allclose
    assert np.allclose(s, np.zeros_like(s))


def test_pload_glob():
    """ Test that pload works on glob-like patterns """
    stream = pload(Path(__file__).parent / "data" / "test_data*.npy", load_func=np.load)
    s = last(isum(stream)).astype(float)  # Cast to float for np.allclose
    assert np.allclose(s, np.zeros_like(s))

    stream = pload(
        Path(__file__).parent / "data" / "test_data*.npy",
        load_func=np.load,
        processes=2,
    )
    s = last(isum(stream)).astype(float)  # Cast to float for np.allclose
    assert np.allclose(s, np.zeros_like(s))


def test_pload_file_list():
    """ Test that pload works on iterable of filenames """
    files = [
        Path(__file__).parent / "data" / "test_data1.npy",
        Path(__file__).parent / "data" / "test_data2.npy",
        Path(__file__).parent / "data" / "test_data3.npy",
    ]
    stream = pload(files, load_func=np.load)
    s = last(isum(stream)).astype(float)  # Cast to float for np.allclose
    assert np.allclose(s, np.zeros_like(s))

    files = [
        Path(__file__).parent / "data" / "test_data1.npy",
        Path(__file__).parent / "data" / "test_data2.npy",
        Path(__file__).parent / "data" / "test_data3.npy",
    ]
    stream = pload(files, load_func=np.load, processes=2)
    s = last(isum(stream)).astype(float)  # Cast to float for np.allclose
    assert np.allclose(s, np.zeros_like(s))
