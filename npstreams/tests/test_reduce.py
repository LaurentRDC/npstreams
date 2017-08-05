# -*- coding: utf-8 -*-
import unittest
import numpy as np

from .. import ireduce_ufunc, last

# Only testing binary ufuncs that support floats
# i.e. leaving bitwise_* and logical_* behind
# Also, numpy.ldexp takes in ints and floats separately, so
# i leave it behind
UFUNCS = (np.add, np.subtract, np.multiply, np.divide,
            np.logaddexp, np.logaddexp2, np.true_divide, np.floor_divide,
            np.power, np.remainder, np.mod, np.fmod, np.arctan2,
            np.hypot, np.greater, np.greater_equal, np.less,
            np.less_equal, np.not_equal, np.equal, np.maximum,
            np.fmax, np.minimum, np.fmin, np.copysign,
            np.nextafter)

class TestIreduceUfunc(unittest.TestCase):

    def setUp(self):
        self.source = [np.random.random((16,5,8)) for _ in range(10)]
        self.stack = np.stack(self.source, axis = -1)
    
    def test_no_side_effects(self):
        """ Test that no arrays in the stream are modified """
        for arr in self.source:
            arr.setflags(write = False)
        out = last(ireduce_ufunc(self.source, np.add))
    
    def test_single_array(self):
        """ Test ireduce_ufunc on a single array, not a sequence """
        source = np.ones( (16, 16), dtype = np.int)
        out = last(ireduce_ufunc(source, np.add, axis = -1))
        self.assertTrue(np.allclose(source, out))
    
    def test_non_ufunc(self):
        """ Test that ireduce_ufunc raises TypeError when a non-binary ufunc is passed """
        with self.assertRaises(TypeError):
            ireduce_ufunc(range(10), ufunc = lambda x: x)
        
    def test_output_shape(self):
        """ Test output shape """
        for axis in (0, 1, 2, 3, None):
            with self.subTest('axis = {}'.format(axis)):
                from_numpy = np.add.reduce(self.stack, axis = axis)
                out = last(ireduce_ufunc(self.source, np.add, axis = axis))
                self.assertSequenceEqual(from_numpy.shape, out.shape)
                self.assertTrue(np.allclose(out, from_numpy))

    def test_length(self):
        """ Test that the number of elements yielded by ireduce_ufunc is correct """
        for axis in (0, 1, 2, 3, None):
            with self.subTest('axis = {}'.format(axis)):
                source = (np.zeros((16, 5, 8)) for _ in range(10))
                out = list(ireduce_ufunc(source, np.add, axis = axis))
                self.assertEqual(10, len(out))

# Dynamics generation of tests on binary ufuncs
def test_binary_ufunc(ufunc):
    """ Generate a test to ensure that ireduce_ufunc(..., ufunc, ...) 
    works as intendent."""
    def test_ufunc(self):
        def sufunc(arrays, axis = -1):  #s for stream
            return last(ireduce_ufunc(arrays, ufunc, axis = axis))
        for axis in (0, 1, 2, -1):
            from_numpy = ufunc.reduce(self.stack, axis = axis)
            from_sufunc = sufunc(self.source, axis = axis)
            self.assertSequenceEqual(from_sufunc.shape, from_numpy.shape)
            self.assertTrue(np.allclose(from_numpy, from_sufunc))
    return test_ufunc

class TestAllBinaryUfuncs(unittest.TestCase):

    def setUp(self):
        self.source = [np.random.random((16,5,8)) for _ in range(10)]
        self.stack = np.stack(self.source, axis = -1)

for ufunc in UFUNCS:
    test_name = 'test_ireduce_ufunc_on_{}'.format(ufunc.__name__)
    test = test_binary_ufunc(ufunc)
    setattr(TestAllBinaryUfuncs, test_name, test)

if __name__ == '__main__':
    unittest.main()