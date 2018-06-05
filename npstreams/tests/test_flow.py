# -*- coding: utf-8 -*-
import unittest
import numpy as np

from .. import array_stream, ipipe, last, iload, pload, isum

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

class TestILoad(unittest.TestCase):

    def test_glob(self):
        """ Test that iload works on glob-like patterns """
        stream = iload('npstreams\\tests\\data\\test_data*.npy', load_func = np.load)
        s = last(isum(stream)).astype(np.float)     # Cast to float for np.allclose
        self.assertTrue(np.allclose(s, np.zeros_like(s)))
    
    def test_file_list(self):
        """ Test that iload works on iterable of filenames """
        files = ['npstreams\\tests\\data\\test_data1.npy',
                 'npstreams\\tests\\data\\test_data2.npy',
                 'npstreams\\tests\\data\\test_data3.npy']
        stream = iload(files, load_func = np.load)
        s = last(isum(stream)).astype(np.float)     # Cast to float for np.allclose
        self.assertTrue(np.allclose(s, np.zeros_like(s)))

class TestPLoad(unittest.TestCase):

    def test_glob(self):
        """ Test that pload works on glob-like patterns """
        with self.subTest('processes = 1'):
            stream = pload('npstreams\\tests\\data\\test_data*.npy', load_func = np.load)
            s = last(isum(stream)).astype(np.float)     # Cast to float for np.allclose
            self.assertTrue(np.allclose(s, np.zeros_like(s)))

        with self.subTest('processes = 2'):
            stream = pload('npstreams\\tests\\data\\test_data*.npy', load_func = np.load, processes = 2)
            s = last(isum(stream)).astype(np.float)     # Cast to float for np.allclose
            self.assertTrue(np.allclose(s, np.zeros_like(s)))
    
    def test_file_list(self):
        """ Test that pload works on iterable of filenames """
        with self.subTest('processes = 1'):
            files = ['npstreams\\tests\\data\\test_data1.npy',
                    'npstreams\\tests\\data\\test_data2.npy',
                    'npstreams\\tests\\data\\test_data3.npy']
            stream = pload(files, load_func = np.load)
            s = last(isum(stream)).astype(np.float)     # Cast to float for np.allclose
            self.assertTrue(np.allclose(s, np.zeros_like(s)))

        with self.subTest('processes = 2'):
            files = ['npstreams\\tests\\data\\test_data1.npy',
                    'npstreams\\tests\\data\\test_data2.npy',
                    'npstreams\\tests\\data\\test_data3.npy']
            stream = pload(files, load_func = np.load, processes = 2)
            s = last(isum(stream)).astype(np.float)     # Cast to float for np.allclose
            self.assertTrue(np.allclose(s, np.zeros_like(s)))

if __name__ == '__main__':
    unittest.main()