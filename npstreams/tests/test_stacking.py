# -*- coding: utf-8 -*-
import unittest
import numpy as np

from .. import last, istack

class TestIStack(unittest.TestCase):

    def test_against_numpy_stack(self):
        """ Test against numpy.stack for axis = -1 and """
        stream = [np.random.random((15,7,2,1)) for _ in range(10)]
        with self.subTest('axis = -1'):
            stack = np.stack(stream, axis = -1)
            from_istack = last(istack(stream, axis = -1))
            self.assertSequenceEqual(stack.shape, from_istack.shape)

    def test_against_numpy_concatenate(self):
        """ Test against numpy.concatenate for existing axes """
        stream = [np.random.random((15,7,2,1)) for _ in range(10)]
        for axis in range(4):
            with self.subTest('axis = {}'.format(axis)):
                stack = np.concatenate(stream, axis = axis)
                from_istack = last(istack(stream, axis = axis))
                self.assertSequenceEqual(stack.shape, from_istack.shape)

if __name__ == '__main__':
	unittest.main()
