# -*- coding: utf-8 -*-
import unittest
import numpy as np

from npstreams import ireduce_ufunc, preduce_ufunc, last, nan_to_num, reduce_ufunc

# Only testing binary ufuncs that support floats
# i.e. leaving bitwise_* and logical_* behind
# Also, numpy.ldexp takes in ints and floats separately, so
# leave it behind
UFUNCS = (
    np.add,
    np.subtract,
    np.multiply,
    np.divide,
    np.logaddexp,
    np.logaddexp2,
    np.true_divide,
    np.floor_divide,
    np.power,
    np.remainder,
    np.mod,
    np.fmod,
    np.arctan2,
    np.hypot,
    np.maximum,
    np.fmax,
    np.minimum,
    np.fmin,
    np.copysign,
    np.nextafter,
)


class TestIreduceUfunc(unittest.TestCase):
    def setUp(self):
        self.source = [np.random.random((16, 5, 8)) for _ in range(10)]
        self.stack = np.stack(self.source, axis=-1)

    def test_no_side_effects(self):
        """ Test that no arrays in the stream are modified """
        for arr in self.source:
            arr.setflags(write=False)
        out = last(ireduce_ufunc(self.source, np.add))

    def test_single_array(self):
        """ Test ireduce_ufunc on a single array, not a sequence """
        source = np.ones((16, 16), dtype=np.int)
        out = last(ireduce_ufunc(source, np.add, axis=-1))
        self.assertTrue(np.allclose(source, out))

    def test_out_parameter(self):
        """ Test that the kwargs ``out`` is correctly passed to reduction function """

        with self.subTest("axis = -1"):
            not_out = last(ireduce_ufunc(self.source, np.add, axis=-1))
            out = np.empty_like(self.source[0])
            last(ireduce_ufunc(self.source, ufunc=np.add, out=out))

            self.assertTrue(np.allclose(not_out, out))

        with self.subTest("axis != -1"):
            not_out = last(ireduce_ufunc(self.source, np.add, axis=2))
            out = np.empty_like(self.source[0])
            from_out = last(ireduce_ufunc(self.source, ufunc=np.add, out=out, axis=2))

            self.assertTrue(np.allclose(not_out, from_out))

    def test_ignore_nan_no_identity(self):
        """ Test ireduce_ufunc on an ufunc with no identity raises
        an error for ignore_nan = True """
        source = [np.ones((16, 16), dtype=np.int) for _ in range(5)]
        with self.assertRaises(ValueError):
            ireduce_ufunc(source, np.maximum, axis=-1, ignore_nan=True)

    def test_non_ufunc(self):
        """ Test that ireduce_ufunc raises TypeError when a non-ufunc is passed """
        with self.assertRaises(TypeError):
            ireduce_ufunc(range(10), ufunc=lambda x: x)

    def test_non_binary_ufunc(self):
        """ Test that ireduce_ufunc raises ValueError if non-binary ufunc is used """
        with self.assertRaises(ValueError):
            ireduce_ufunc(range(10), ufunc=np.absolute)

    def test_output_shape(self):
        """ Test output shape """
        for axis in (0, 1, 2, 3, None):
            with self.subTest("axis = {}".format(axis)):
                from_numpy = np.add.reduce(self.stack, axis=axis)
                out = last(ireduce_ufunc(self.source, np.add, axis=axis))
                self.assertSequenceEqual(from_numpy.shape, out.shape)
                self.assertTrue(np.allclose(out, from_numpy))

    def test_length(self):
        """ Test that the number of elements yielded by ireduce_ufunc is correct """
        for axis in (0, 1, 2, 3, None):
            with self.subTest("axis = {}".format(axis)):
                source = (np.zeros((16, 5, 8)) for _ in range(10))
                out = list(ireduce_ufunc(source, np.add, axis=axis))
                self.assertEqual(10, len(out))

    def test_ignore_nan(self):
        """ Test that ignore_nan is working """
        for axis in (0, 1, 2, 3, None):
            with self.subTest("axis = {}".format(axis)):
                out = last(
                    ireduce_ufunc(self.source, np.add, axis=axis, ignore_nan=True)
                )
                self.assertFalse(np.any(np.isnan(out)))


class TestPreduceUfunc(unittest.TestCase):
    def test_trivial(self):
        """ Test preduce_ufunc for a sum of zeroes over two processes"""
        stream = [np.zeros((8, 8)) for _ in range(10)]
        s = preduce_ufunc(stream, ufunc=np.add, processes=2, ntotal=10)
        self.assertTrue(np.allclose(s, np.zeros_like(s)))

    def test_correctess(self):
        """ Test preduce_ufunc is equivalent to reduce_ufunc for random sums"""
        stream = [np.random.random((8, 8)) for _ in range(20)]
        s = preduce_ufunc(stream, ufunc=np.add, processes=3, ntotal=10)
        self.assertTrue(np.allclose(s, reduce_ufunc(stream, np.add)))


# Dynamics generation of tests on binary ufuncs
def test_binary_ufunc(ufunc):
    """ Generate a test to ensure that ireduce_ufunc(..., ufunc, ...) 
    works as intendent."""

    def test_ufunc(self):
        def sufunc(arrays, axis=-1):  # s for stream
            return last(ireduce_ufunc(arrays, ufunc, axis=axis))

        for axis in (0, 1, 2, -1):
            with self.subTest("axis = {}".format(axis)):
                from_numpy = ufunc.reduce(self.stack, axis=axis)
                from_sufunc = sufunc(self.source, axis=axis)
                self.assertSequenceEqual(from_sufunc.shape, from_numpy.shape)
                self.assertTrue(np.allclose(from_numpy, from_sufunc))

    return test_ufunc


class TestAllBinaryUfuncs(unittest.TestCase):
    def setUp(self):
        self.source = [np.random.random((16, 5, 8)) for _ in range(10)]
        self.stack = np.stack(self.source, axis=-1)


for ufunc in UFUNCS:
    test_name = "test_ireduce_ufunc_on_{}".format(ufunc.__name__)
    test = test_binary_ufunc(ufunc)
    setattr(TestAllBinaryUfuncs, test_name, test)


def test_binary_ufunc_ignore_nan(ufunc):
    """ Generate a test to ensure that ireduce_ufunc(..., ufunc, ...) 
    works as intendent with NaNs in stream."""

    def test_ufunc(self):
        stack = nan_to_num(self.stack, fill_value=ufunc.identity)

        def sufunc(arrays, ignore_nan=False):  # s for stream
            return last(ireduce_ufunc(arrays, ufunc, axis=1, ignore_nan=True))

        from_numpy = ufunc.reduce(stack, axis=1)
        from_sufunc = sufunc(self.source)
        self.assertSequenceEqual(from_numpy.shape, from_sufunc.shape)
        self.assertTrue(np.allclose(from_numpy, from_sufunc))

    return test_ufunc


class TestAllBinaryUfuncsIgnoreNans(unittest.TestCase):
    def setUp(self):
        self.source = [np.random.random((16, 5, 8)) for _ in range(10)]
        self.source[0][0, 0, 0] = np.nan
        self.stack = np.stack(self.source, axis=-1)


for ufunc in UFUNCS:
    if ufunc.identity is None:
        continue
    test_name = "test_ireduce_ufunc_on_{}".format(ufunc.__name__)
    test = test_binary_ufunc_ignore_nan(ufunc)
    setattr(TestAllBinaryUfuncsIgnoreNans, test_name, test)

if __name__ == "__main__":
    unittest.main()
