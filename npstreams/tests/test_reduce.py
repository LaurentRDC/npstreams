# -*- coding: utf-8 -*-
import unittest
import numpy as np

from .. import stream_reduce, last, iall, iany

class TestStreamReduce(unittest.TestCase):

    def setUp(self):
        self.source = [np.random.random((16,5,8)) for _ in range(10)]
        self.stack = np.stack(self.source, axis = -1)
    
    def test_no_side_effects(self):
        """ Test that no arrays in the stream are modified """
        for arr in self.source:
            arr.setflags(write = False)
        out = last(stream_reduce(self.source, np.sum))
    
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

class TestIAll(unittest.TestCase):

    def test_against_numpy(self):
        """ Test iall against numpy.all """
        stream = [np.zeros((8, 16, 2)) for _ in range(11)]
        stream[3][3,0,1] = 1    # so that np.all(axis = None) evaluates to False
        stack = np.stack(stream, axis = -1)

        with self.subTest('axis = None'):
            from_numpy = np.all(stack, axis = None)
            from_stream = last(iall(stream, axis = None))
            self.assertEqual(from_numpy, from_stream)

        for axis in range(stack.ndim):
            with self.subTest('axis = {}'.format(axis)):
                from_numpy = np.all(stack, axis = axis)
                from_stream = last(iall(stream, axis = axis))
                self.assertTrue(np.allclose(from_numpy, from_stream))

class TestIAny(unittest.TestCase):

    def test_against_numpy(self):
        """ Test iany against numpy.any """
        stream = [np.zeros((8, 16, 2)) for _ in range(11)]
        stream[3][3,0,1] = 1    # so that np.all(axis = None) evaluates to False
        stack = np.stack(stream, axis = -1)

        with self.subTest('axis = None'):
            from_numpy = np.any(stack, axis = None)
            from_stream = last(iany(stream, axis = None))
            self.assertEqual(from_numpy, from_stream)

        for axis in range(stack.ndim):
            with self.subTest('axis = {}'.format(axis)):
                from_numpy = np.any(stack, axis = axis)
                from_stream = last(iany(stream, axis = axis))
                self.assertTrue(np.allclose(from_numpy, from_stream))

if __name__ == '__main__':
	unittest.main()
