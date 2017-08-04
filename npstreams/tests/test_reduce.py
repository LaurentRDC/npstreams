# -*- coding: utf-8 -*-
import unittest
import numpy as np

from .. import stream_ufunc, last

class TestStreamReduce(unittest.TestCase):

    def setUp(self):
        self.source = [np.random.random((16,5,8)) for _ in range(10)]
        self.stack = np.stack(self.source, axis = -1)
    
    def test_no_side_effects(self):
        """ Test that no arrays in the stream are modified """
        for arr in self.source:
            arr.setflags(write = False)
        out = last(stream_ufunc(self.source, np.add))
    
    def test_single_array(self):
        """ Test stream_ufunc on a single array, not a sequence """
        source = np.ones( (16, 16), dtype = np.int)
        out = last(stream_ufunc(source, np.add, axis = -1))
        self.assertTrue(np.allclose(source, out))
    
    def test_nonbinary_ufunc(self):
        """ Test that stream_ufunc raises TypeError when a non-binary ufunc is passed """
        with self.assertRaises(TypeError):
            next(stream_ufunc(range(10), ufunc = np.sqrt))
    
    def test_non_ufunc(self):
        """ Test that stream_ufunc raises TypeError when a non-binary ufunc is passed """
        with self.assertRaises(TypeError):
            next(stream_ufunc(range(10), ufunc = lambda x: x))
        
    def test_output_shape(self):
        """ Test output shape """
        for axis in (0, 1, 2, 3, None):
            with self.subTest('axis = {}'.format(axis)):
                from_numpy = np.add.reduce(self.stack, axis = axis)
                out = last(stream_ufunc(self.source, np.add, axis = axis))
                self.assertSequenceEqual(from_numpy.shape, out.shape)
                self.assertTrue(np.allclose(out, from_numpy))

    def test_length(self):
        """ Test that the number of elements yielded by stream_ufunc is correct """
        for axis in (0, 1, 2, 3, None):
            with self.subTest('axis = {}'.format(axis)):
                source = (np.zeros((16, 5, 8)) for _ in range(10))
                out = list(stream_ufunc(source, np.add, axis = axis))
                self.assertEqual(10, len(out))

if __name__ == '__main__':
	unittest.main()
