# -*- coding: utf-8 -*-
import unittest
import numpy as np

from .. import stack

class TestStack(unittest.TestCase):

    def test_against_numpy_stack(self):
        """ Test against numpy.stack for axis = -1 and """
        stream = [np.random.random((15,7,2,1)) for _ in range(10)]
        with self.subTest('axis = -1'):
            dense = np.stack(stream, axis = -1)
            from_stack = stack(stream, axis = -1)
            self.assertTrue(np.allclose(dense, from_stack))

    def test_on_single_array(self):
        """ Test that npstreams.stack works with a single array """
        arr = np.random.random((16,16))
        stacked = stack(arr)
        self.assertTrue(np.allclose(arr[..., np.newaxis], stacked))

    def test_against_numpy_concatenate(self):
        """ Test against numpy.concatenate for existing axes """
        stream = [np.random.random((15,7,2,1)) for _ in range(10)]
        for axis in range(4):
            with self.subTest('axis = {}'.format(axis)):
                dense = np.concatenate(stream, axis = axis)
                from_stack = stack(stream, axis = axis)
                self.assertTrue(np.allclose(dense, from_stack))

if __name__ == '__main__':
	unittest.main()
