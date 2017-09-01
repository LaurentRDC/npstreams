import unittest
import numpy as np
from .. import nan_to_num

class TestNanToNum(unittest.TestCase):

    def test_generic(self):
        """ Test that NaNs are replaced with a fill value """
        with np.errstate(divide='ignore', invalid='ignore'):
            vals = nan_to_num(np.array([0])/0., fill_value = 14)
        self.assertEqual(vals[0], 14)

    def test_integer(self):
        """ Test that nan_to_num on integers does nothing """
        vals = nan_to_num(1)
        self.assertEqual(vals, 1)
        vals = nan_to_num([1])
        self.assertTrue(np.allclose(vals, np.array([1])))

    def test_complex_good(self):
        """ Test nan_to_num on complex input """
        vals = nan_to_num(1+1j)
        self.assertEqual(vals, 1+1j)

if __name__ == '__main__':
    unittest.main()