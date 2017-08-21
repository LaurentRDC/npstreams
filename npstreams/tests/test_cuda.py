# -*- coding: utf-8 -*-

import numpy as np
import unittest
try:
    import pycuda
    WITH_CUDA = True

    from ..cuda import csum
except ImportError:
    WITH_CUDA = False

@unittest.skipIf(not WITH_CUDA, 'pycuda is not installed/available')
class TestCSum(unittest.TestCase):

    def test_zero_sum(self):
        stream = repeat(np.zeros( (64,64), dtype = np.float ), times = 5)
        s = csum(stream)
        self.assertTrue(np.allclose(s, np.zeros((64,64))))

if __name__ == '__main__':
    unittest.main()