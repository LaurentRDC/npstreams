# -*- coding: utf-8 -*-
import unittest
from random import randint, random

import numpy as np

from .. import isum, inansum, iprod, inanprod, last

class TestISum(unittest.TestCase):

    def test_trivial(self):
        """ Test a sum of zeros """
        source = [np.zeros((16,), dtype = np.float) for _ in range(10)]
        summed = last(isum(source))
        self.assertTrue(np.allclose(summed, np.zeros_like(summed)))
    
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
