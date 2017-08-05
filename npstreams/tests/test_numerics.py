# -*- coding: utf-8 -*-
import unittest
from random import randint, random

import numpy as np

from .. import (isum, inansum, psum, iprod, pprod, inanprod, 
                last, isub, iany, iall)

class TestISum(unittest.TestCase):

    def test_trivial(self):
        """ Test a sum of zeros """
        source = [np.zeros((16,), dtype = np.float) for _ in range(10)]
        summed = last(isum(source))
        self.assertTrue(np.allclose(summed, np.zeros_like(summed)))

    def test_ignore_nans(self):
        """ Test a sum of zeros with NaNs sprinkled """
        source = [np.zeros((16,), dtype = np.float) for _ in range(10)]
        source.append(np.full((16,), fill_value = np.nan))
        summed = last(isum(source, ignore_nan = True))
        self.assertTrue(np.allclose(summed, np.zeros_like(summed)))
    
    def test_length(self):
        """ Test that the number of yielded elements is the same as source """
        source = [np.zeros((16,), dtype = np.float) for _ in range(10)]
        summed = list(isum(source, axis = 0))
        self.assertEqual(10, len(summed))

    def test_dtype(self):
        """ Test a sum of floating zeros with an int accumulator """
        source = [np.zeros((16,), dtype = np.float) for _ in range(10)]
        summed = last(isum(source, dtype = np.int))
        self.assertTrue(np.allclose(summed, np.zeros_like(summed)))
        self.assertEqual(summed.dtype, np.int)
    
    def test_axis(self):
        """ Test that isum(axis = 0) yields 0d arrays """
        source = [np.zeros((16,), dtype = np.float) for _ in range(10)]
        
        with self.subTest('axis = 0'):
            summed = last(isum(source, axis = 0))
            self.assertTrue(np.allclose(summed, np.zeros_like(summed)))

        with self.subTest('axis = None'):
            summed = last(isum(source, axis = None))
            self.assertTrue(np.allclose(summed, 0))
    
    def test_return_shape(self):
        """ Test that the shape of output is as expected """
        source = [np.zeros((16,), dtype = np.float) for _ in range(10)]

        with self.subTest('axis = 0'):
            summed = last(isum(source, axis = 0))
            self.assertSequenceEqual(summed.shape, (1,10))
    
    #@unittest.skip('')
    def test_against_numpy(self):
        """ Test that isum() returns the same as numpy.sum() for various axis inputs """

        stream = [np.random.random((16,16)) for _ in range(10)]
        stack = np.dstack(stream)

        for axis in (0, 1, 2, None):
            with self.subTest('axis = {}'.format(axis)):
                from_numpy = np.sum(stack, axis = axis)
                from_isum = last(isum(stream, axis = axis))
                self.assertTrue(np.allclose(from_isum, from_numpy))

class TestPSum(unittest.TestCase):

    def test_trivial(self):
        """ Test a sum of zeros """
        source = [np.zeros((16,), dtype = np.float) for _ in range(10)]
        summed = psum(source)
        self.assertTrue(np.allclose(summed, np.zeros_like(summed)))

    def test_ignore_nans(self):
        """ Test a sum of zeros with NaNs sprinkled """
        source = [np.zeros((16,), dtype = np.float) for _ in range(10)]
        source.append(np.full((16,), fill_value = np.nan))
        summed = psum(source, ignore_nan = True)
        self.assertTrue(np.allclose(summed, np.zeros_like(summed)))

    def test_dtype(self):
        """ Test a sum of floating zeros with an int accumulator """
        source = [np.zeros((16,), dtype = np.float) for _ in range(10)]
        summed = psum(source, dtype = np.int)
        self.assertTrue(np.allclose(summed, np.zeros_like(summed)))
        self.assertEqual(summed.dtype, np.int)

    def test_axis(self):
        """ Test that psum(axis = 0) yields 0d arrays """
        source = [np.zeros((16,), dtype = np.float) for _ in range(10)]
        
        with self.subTest('axis = 0'):
            summed = psum(source, axis = 0)
            self.assertTrue(np.allclose(summed, np.zeros_like(summed)))

        with self.subTest('axis = None'):
            summed = psum(source, axis = None)
            self.assertTrue(np.allclose(summed, 0))
    
class TestINanSum(unittest.TestCase):
    
    def test_trivial(self):
        """ Test a sum of zeros """
        source = [np.zeros((16,), dtype = np.float) for _ in range(10)]
        summed = last(inansum(source))
        self.assertTrue(np.allclose(summed, np.zeros_like(summed)))

class TestIProd(unittest.TestCase):

    def test_trivial(self):
        """ Test a product of ones """
        source = [np.ones((16,), dtype = np.float) for _ in range(10)]
        product = last(iprod(source))
        self.assertTrue(np.allclose(product, np.ones_like(product)))

    def test_ignore_nans(self):
        """ Test that NaNs are ignored. """
        source = [np.ones((16,), dtype = np.float) for _ in range(10)]
        source.append(np.full_like(source[0], np.nan))
        product = last(iprod(source, ignore_nan = True))
        self.assertTrue(np.allclose(product, np.ones_like(product)))

    def test_dtype(self):
        """ Test that dtype argument is working """
        source = [np.ones((16,), dtype = np.float) for _ in range(10)]
        product = last(iprod(source, dtype = np.int))
        self.assertTrue(np.allclose(product, np.ones_like(product)))
        self.assertEqual(product.dtype, np.int)

    def test_axis(self):
        """ Test that iprod(axis = 0) yields 0d arrays """
        source = [np.ones((16,), dtype = np.float) for _ in range(10)]
        
        with self.subTest('axis = 0'):
            summed = last(iprod(source, axis = 0))
            self.assertTrue(np.all(summed == 1))

        with self.subTest('axis = None'):
            summed = last(iprod(source, axis = None))
            self.assertTrue(np.allclose(summed, np.ones_like(summed)))

    #@unittest.skip('')
    def test_against_numpy(self):
        """ Test that iprod() returns the same as numpy.prod() for various axis inputs """

        stream = [np.random.random((16,16)) for _ in range(10)]
        stack = np.dstack(stream)

        for axis in (0, 1, 2, None):
            with self.subTest('axis = {}'.format(axis)):
                from_numpy = np.prod(stack, axis = axis)
                from_stream = last(iprod(stream, axis = axis))
                self.assertTrue(np.allclose(from_stream, from_numpy))

class TestPProd(unittest.TestCase):

    def test_trivial(self):
        """ Test a product of ones """
        source = [np.ones((16,), dtype = np.float) for _ in range(10)]
        product = pprod(source)
        self.assertTrue(np.allclose(product, np.ones_like(product)))
    
    def test_ignore_nans(self):
        """ Test that NaNs are ignored. """
        source = [np.ones((16,), dtype = np.float) for _ in range(10)]
        source.append(np.full_like(source[0], np.nan))
        product = pprod(source, ignore_nan = True)
        self.assertTrue(np.allclose(product, np.ones_like(product)))

    def test_dtype(self):
        """ Test that dtype argument is working """
        source = [np.ones((16,), dtype = np.float) for _ in range(10)]
        product = pprod(source, dtype = np.int)
        self.assertTrue(np.allclose(product, np.ones_like(product)))
        self.assertEqual(product.dtype, np.int)

    def test_axis(self):
        """ Test that iprod(axis = 0) yields 0d arrays """
        source = [np.ones((16,), dtype = np.float) for _ in range(10)]
        
        with self.subTest('axis = 0'):
            summed = pprod(source, axis = 0)
            self.assertTrue(np.all(summed == 1))

        with self.subTest('axis = None'):
            summed = pprod(source, axis = None)
            self.assertTrue(summed, np.ones_like(summed))
        
    
class TestINanProd(unittest.TestCase):
    
    def test_trivial(self):
        """ Test a product of ones """
        source = [np.ones((16,), dtype = np.float) for _ in range(10)]
        product = last(inanprod(source))
        self.assertTrue(np.allclose(product, np.ones_like(product)))

class TestISub(unittest.TestCase):
    
    def test_against_numpy(self):
        """ Test against numpy.subtract.reduce """
        stream = [np.random.random((8, 16, 2)) for _ in range(11)]
        stack = np.stack(stream, axis = -1)

        for axis in range(stack.ndim):
            with self.subTest('axis = {}'.format(axis)):
                from_numpy = np.subtract.reduce(stack, axis = axis)
                from_stream = last(isub(stream, axis = axis))
                self.assertTrue(np.allclose(from_numpy, from_stream))

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
