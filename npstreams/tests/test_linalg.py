# -*- coding: utf-8 -*-
import unittest
from random import randint, random

import numpy as np

from .. import idot, itensordot, last

class TestIDot(unittest.TestCase):

    def test_against_numpy_multidot(self):
        """ Test iany against numpy.linalg.multi_dot in 2D case """
        stream = [np.random.random((8, 8)) for _ in range(7)]

        from_numpy = np.linalg.multi_dot(stream)
        from_stream = last(idot(stream))

        self.assertSequenceEqual(from_numpy.shape, from_stream.shape)
        self.assertTrue(np.allclose(from_numpy, from_stream))

class TestITensordot(unittest.TestCase):

    def test_against_numpy_tensordot(self):
        """ Test iany against numpy.linalg.multi_dot in 2D case """
        stream = tuple(np.random.random((8, 8)) for _ in range(2))

        for axis in (0, 1, 2):
            with self.subTest('axis = {}'.format(axis)):
                from_numpy = np.tensordot(*stream)
                from_stream = last(itensordot(stream))

                self.assertSequenceEqual(from_numpy.shape, from_stream.shape)
                self.assertTrue(np.allclose(from_numpy, from_stream))

if __name__ == '__main__':
	unittest.main()
