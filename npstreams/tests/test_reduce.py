# -*- coding: utf-8 -*-
import unittest
import numpy as np

from .. import stream_reduce, last

class TestStreamReduce(unittest.TestCase):

    def setUp(self):
        self.source = [np.random.random((16,5,8)) for _ in range(10)]
        self.stack = np.stack(self.source, axis = -1)
    
    def test_no_side_effects(self):
        """ Test that no arrays in the stream are modified """
        for arr in self.source:
            arr.setflags(write = False)
        out = last(stream_reduce(self.source, np.sum))
    
    def test_single_array(self):
        """ Test stream_reduce on a single array, not a sequence """
        source = np.ones( (16, 16), dtype = np.int)
        out = last(stream_reduce(source, np.sum, axis = -1))
        self.assertTrue(np.allclose(source, out))
        
    def test_output_shape(self):
        """ Test output shape """
        for axis in (0, 1, 2, 3, None):
            with self.subTest('axis = {}'.format(axis)):
                from_numpy = np.add.reduce(self.stack, axis = axis)
                out = last(stream_reduce(self.source, np.add.reduce, axis = axis))
                self.assertSequenceEqual(from_numpy.shape, out.shape)
                self.assertTrue(np.allclose(out, from_numpy))
    
    def test_length(self):
        """ Test that the number of elements yielded by stream_reduce is correct """
        for axis in (0, 1, 2, 3, None):
            with self.subTest('axis = {}'.format(axis)):
                source = (np.zeros((16, 5, 8)) for _ in range(10))
                out = list(stream_reduce(source, np.sum, axis = axis))
                self.assertEqual(10, len(out))

if __name__ == '__main__':
	unittest.main()
