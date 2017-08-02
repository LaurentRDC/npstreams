# -*- coding: utf-8 -*-
import unittest
from random import randint, random

import numpy as np

from .. import isum, inansum, psum, iprod, pprod, inanprod, last

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
            self.assertEqual(summed, 0)

        with self.subTest('axis = None'):
            summed = last(isum(source, axis = None))
            self.assertEqual(summed, 0)

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
            self.assertEqual(summed, 0)

        with self.subTest('axis = None'):
            summed = psum(source, axis = None)
            self.assertEqual(summed, 0)
    
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
            self.assertEqual(summed, 1)

        with self.subTest('axis = None'):
            summed = last(iprod(source, axis = None))
            self.assertEqual(summed, 1)

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
            self.assertEqual(summed, 1)

        with self.subTest('axis = None'):
            summed = pprod(source, axis = None)
            self.assertEqual(summed, 1)
        
    
class TestINanProd(unittest.TestCase):
    
    def test_trivial(self):
        """ Test a product of ones """
        source = [np.ones((16,), dtype = np.float) for _ in range(10)]
        product = last(inanprod(source))
        self.assertTrue(np.allclose(product, np.ones_like(product)))

if __name__ == '__main__':
	unittest.main()
