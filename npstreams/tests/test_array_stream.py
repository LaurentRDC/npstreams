# -*- coding: utf-8 -*-
import unittest
import numpy as np

from .. import array_stream, ipipe, iaverage, last

@array_stream
def iden(arrays):
    yield from arrays

class TestIPipe(unittest.TestCase):
    
    def test_order(self):
        """ Test that ipipe(f, g, h, arrays) -> f(g(h(arr))) for arr in arrays """
        stream = [np.random.random((15,7,2,1)) for _ in range(10)]
        squared = [np.cbrt(np.square(arr)) for arr in stream]
        pipeline = ipipe(np.cbrt, np.square, stream)

        self.assertTrue(all(np.allclose(s, p) for s, p in zip(pipeline, squared)))

    def test_multiprocessing(self):
        """ Test that ipipe(f, g, h, arrays) -> f(g(h(arr))) for arr in arrays """
        stream = [np.random.random((15,7,2,1)) for _ in range(10)]
        squared = [np.cbrt(np.square(arr)) for arr in stream]
        pipeline = ipipe(np.cbrt, np.square, stream, processes = 2)

        self.assertTrue(all(np.allclose(s, p) for s, p in zip(pipeline, squared)))

class TestArrayStream(unittest.TestCase):

    def test_type(self):
        """ Test that all object from an array stream are ndarrays """
        stream = [0, 1, np.array([1])]
        for arr in iden(stream):
            self.assertIsInstance(arr, np.ndarray)

if __name__ == '__main__':
	unittest.main()
