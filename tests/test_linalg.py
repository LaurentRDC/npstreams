# -*- coding: utf-8 -*-
import unittest
from random import randint, random

import numpy as np

from npstreams import idot, itensordot, iinner, ieinsum, last


class TestIDot(unittest.TestCase):
    def test_against_numpy_multidot(self):
        """ Test against numpy.linalg.multi_dot in 2D case """
        stream = [np.random.random((8, 8)) for _ in range(7)]

        from_numpy = np.linalg.multi_dot(stream)
        from_stream = last(idot(stream))

        self.assertSequenceEqual(from_numpy.shape, from_stream.shape)
        self.assertTrue(np.allclose(from_numpy, from_stream))


class TestITensordot(unittest.TestCase):
    def test_against_numpy_tensordot(self):
        """ Test against numpy.tensordot in 2D case """
        stream = tuple(np.random.random((8, 8)) for _ in range(2))

        for axis in (0, 1, 2):
            with self.subTest("axis = {}".format(axis)):
                from_numpy = np.tensordot(*stream)
                from_stream = last(itensordot(stream))

                self.assertSequenceEqual(from_numpy.shape, from_stream.shape)
                self.assertTrue(np.allclose(from_numpy, from_stream))


class TestIInner(unittest.TestCase):
    def test_against_numpy_inner(self):
        """ Test against numpy.tensordot in 2D case """
        stream = tuple(np.random.random((8, 8)) for _ in range(2))

        for axis in (0, 1, 2):
            with self.subTest("axis = {}".format(axis)):
                from_numpy = np.inner(*stream)
                from_stream = last(iinner(stream))

                self.assertSequenceEqual(from_numpy.shape, from_stream.shape)
                self.assertTrue(np.allclose(from_numpy, from_stream))


class TestIEinsum(unittest.TestCase):
    def test_against_numpy_einsum(self):
        """ Test against numpy.einsum  """
        a = np.arange(60.0).reshape(3, 4, 5)
        b = np.arange(24.0).reshape(4, 3, 2)
        stream = [a, b]

        from_numpy = np.einsum("ijk,jil->kl", a, b)
        from_stream = last(ieinsum(stream, "ijk,jil->kl"))

        self.assertSequenceEqual(from_numpy.shape, from_stream.shape)
        self.assertTrue(np.allclose(from_numpy, from_stream))


if __name__ == "__main__":
    unittest.main()
