# -*- coding: utf-8 -*-
import unittest
from random import randint, random

import numpy as np

from .. import isum, inansum, psum, iprod, inanprod, last

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
    
class TestINanProd(unittest.TestCase):
    
    def test_trivial(self):
        """ Test a product of ones """
        source = [np.ones((16,), dtype = np.float) for _ in range(10)]
        product = last(inanprod(source))
        self.assertTrue(np.allclose(product, np.ones_like(product)))

if __name__ == '__main__':
	unittest.main()
