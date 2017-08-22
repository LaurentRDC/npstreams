# -*- coding: utf-8 -*-

from itertools import repeat
import numpy as np
import unittest

try:
    from ..cuda import csum, caverage
    WITH_CUDA = True
except ImportError:
    WITH_CUDA = False

@unittest.skipIf(not WITH_CUDA, 'PyCUDA is not installed/available')
class TestCSum(unittest.TestCase):

    def test_zero_sum(self):
        stream = repeat(np.zeros( (64,64), dtype = np.float ), times = 5)
        s = csum(stream)
        self.assertTrue(np.allclose(s, np.zeros((64,64))))

@unittest.skipIf(not WITH_CUDA, 'PyCUDA is not installed/available')
class TestCAverage(unittest.TestCase):

    def test_zero_avg_no_weights(self):
        stream = repeat(np.zeros( (64,64), dtype = np.float ), times = 5)
        s = caverage(stream)
        self.assertTrue(np.allclose(s, np.zeros((64,64))))

if __name__ == '__main__':
    unittest.main()