# -*- coding: utf-8 -*-

from itertools import repeat
import numpy as np
import unittest

try:
    from ..cuda import csum, cprod, caverage, cmean
    WITH_CUDA = True
except ImportError:
    WITH_CUDA = False

@unittest.skipIf(not WITH_CUDA, 'PyCUDA is not installed/available')
class CudaTestCase(unittest.TestCase):
    pass

class TestCSum(CudaTestCase):

    def test_zero_sum(self):
        stream = repeat(np.zeros( (16,16), dtype = np.float ), times = 5)
        s = csum(stream)
        self.assertTrue(np.allclose(s, np.zeros((16,16))))

    def test_dtype(self):
        stream = repeat(np.zeros( (16,16), dtype = np.float ), times = 5)
        s = csum(stream, dtype = np.int16)
        self.assertTrue(np.allclose(s, np.zeros((16,16))))
        self.assertEqual(s.dtype, np.int16)

    def test_ignore_nans(self):
        """ Test a sum of zeros with NaNs sprinkled """
        source = [np.zeros((16,), dtype = np.float) for _ in range(10)]
        source.append(np.full((16,), fill_value = np.nan))
        summed = csum(source, ignore_nan = True)
        self.assertTrue(np.allclose(summed, np.zeros_like(summed)))

class TestCProd(CudaTestCase):

    def test_ones_prod(self):
        stream = repeat(np.ones( (16,16), dtype = np.float ), times = 5)
        s = cprod(stream)
        self.assertTrue(np.allclose(s, np.ones((16,16))))

    def test_ignore_nans(self):
        """ Test that NaNs are ignored. """
        source = [np.ones((16,), dtype = np.float) for _ in range(10)]
        source.append(np.full_like(source[0], np.nan))
        product = cprod(source, ignore_nan = True)
        self.assertTrue(np.allclose(product, np.ones_like(product)))

    def test_dtype(self):
        """ Test that dtype argument is working """
        source = [np.ones((16,), dtype = np.float) for _ in range(10)]
        product = cprod(source, dtype = np.int)
        self.assertTrue(np.allclose(product, np.ones_like(product)))
        self.assertEqual(product.dtype, np.int)

class TestCAverage(CudaTestCase):

    def test_avg_no_weights(self):
        stream = [np.random.random(size = (16,16)) for _ in range(5)]
        from_caverage = caverage(stream)
        from_numpy = np.average(np.dstack(stream), axis = 2)
        self.assertTrue(np.allclose(from_caverage, from_numpy))

    def test_weighted_average(self):
        """ Test results of weighted average against numpy.average """
        stream = [np.random.random(size = (16,16)) for _ in range(5)]
        
        weights = [np.random.random(size = stream[0].shape) for _ in stream]
        from_caverage = caverage(stream, weights = weights)
        from_numpy = np.average(np.dstack(stream), axis = 2, weights = np.dstack(weights))
        self.assertTrue(np.allclose(from_caverage, from_numpy))

class TestCMean(CudaTestCase):

    def test_mean_of_ones(self):
        stream = repeat(np.ones( (16,16), dtype = np.float ), times = 5)
        s = cmean(stream)
        self.assertTrue(np.allclose(s, np.ones((16,16))))

if __name__ == '__main__':
    unittest.main()