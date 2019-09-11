# -*- coding: utf-8 -*-
import unittest
from itertools import repeat
from random import randint, random, seed
from warnings import catch_warnings, simplefilter

import numpy as np

try:
    from scipy.stats import sem as scipy_sem

    WITH_SCIPY = True
except ImportError:
    WITH_SCIPY = False

from npstreams import (
    iaverage,
    imean,
    isem,
    istd,
    ivar,
    last,
    ihistogram,
    mean,
    average,
    sem,
    std,
    var,
)

seed(23)


class TestAverage(unittest.TestCase):
    def test_trivial(self):
        """ Test average() on a stream of zeroes """
        stream = repeat(np.zeros((64, 64), dtype=np.float), times=5)
        for av in average(stream):
            self.assertTrue(np.allclose(av, np.zeros_like(av)))

    def test_vs_numpy(self):
        """ Test average vs. numpy.average """
        stream = [np.random.random(size=(64, 64)) for _ in range(5)]
        stack = np.dstack(stream)

        for axis in (0, 1, 2, None):
            with self.subTest(f"axis = {axis}"):
                from_stream = average(stream, axis=axis)
                from_numpy = np.average(stack, axis=axis)
                self.assertTrue(np.allclose(from_numpy, from_stream))

    def test_weighted_average(self):
        """ Test results of weighted average against numpy.average """
        stream = [np.random.random(size=(16, 16)) for _ in range(5)]

        with self.subTest("float weights"):
            weights = [random() for _ in stream]
            from_average = average(stream, weights=weights)
            from_numpy = np.average(
                np.dstack(stream), axis=2, weights=np.array(weights)
            )
            self.assertTrue(np.allclose(from_average, from_numpy))

        with self.subTest("array weights"):
            weights = [np.random.random(size=stream[0].shape) for _ in stream]
            from_average = average(stream, weights=weights)
            from_numpy = np.average(
                np.dstack(stream), axis=2, weights=np.dstack(weights)
            )
            self.assertTrue(np.allclose(from_average, from_numpy))

    def test_ignore_nan(self):
        """ Test that NaNs are handled correctly """
        stream = [np.random.random(size=(16, 12)) for _ in range(5)]
        for s in stream:
            s[randint(0, 15), randint(0, 11)] = np.nan

        with catch_warnings():
            simplefilter("ignore")
            from_average = average(stream, ignore_nan=True)
        from_numpy = np.nanmean(np.dstack(stream), axis=2)
        self.assertTrue(np.allclose(from_average, from_numpy))


class TestIAverage(unittest.TestCase):
    def test_trivial(self):
        """ Test iaverage on stream of zeroes """
        stream = repeat(np.zeros((64, 64), dtype=np.float), times=5)
        for av in iaverage(stream):
            self.assertTrue(np.allclose(av, np.zeros_like(av)))

    def test_weighted_average(self):
        """ Test results of weighted iverage against numpy.average """
        stream = [np.random.random(size=(16, 16)) for _ in range(5)]

        with self.subTest("float weights"):
            weights = [random() for _ in stream]
            from_iaverage = last(iaverage(stream, weights=weights))
            from_numpy = np.average(
                np.dstack(stream), axis=2, weights=np.array(weights)
            )
            self.assertTrue(np.allclose(from_iaverage, from_numpy))

        with self.subTest("array weights"):
            weights = [np.random.random(size=stream[0].shape) for _ in stream]
            from_iaverage = last(iaverage(stream, weights=weights))
            from_numpy = np.average(
                np.dstack(stream), axis=2, weights=np.dstack(weights)
            )
            self.assertTrue(np.allclose(from_iaverage, from_numpy))

    def test_ignore_nan(self):
        """ Test that NaNs are handled correctly """
        stream = [np.random.random(size=(16, 12)) for _ in range(5)]
        for s in stream:
            s[randint(0, 15), randint(0, 11)] = np.nan

        with catch_warnings():
            simplefilter("ignore")
            from_iaverage = last(iaverage(stream, ignore_nan=True))
        from_numpy = np.nanmean(np.dstack(stream), axis=2)
        self.assertTrue(np.allclose(from_iaverage, from_numpy))

    def test_length(self):
        """ Test that the number of yielded elements is the same as source """
        source = (np.zeros((16,)) for _ in range(5))
        avg = list(iaverage(source, axis=0))
        self.assertEqual(len(avg), 5)

    def test_output_dtype(self):
        """ Test that that yielded arrays are always floats """
        for dtype in (np.uint8, np.bool, np.int16, np.float16):
            with self.subTest(f"Dtype = {dtype}"):
                source = (np.zeros((16,), dtype=dtype) for _ in range(5))
                avg = last(iaverage(source))
                self.assertEqual(avg.dtype, np.float)

    def test_output_shape(self):
        """ Test output shape """
        source = [np.random.random((16, 12, 5)) for _ in range(10)]
        stack = np.stack(source, axis=-1)
        for axis in (0, 1, 2, None):
            with self.subTest(f"axis = {axis}"):
                from_numpy = np.average(stack, axis=axis)
                out = last(iaverage(source, axis=axis))
                self.assertSequenceEqual(from_numpy.shape, out.shape)
                self.assertTrue(np.allclose(out, from_numpy))


class TestMean(unittest.TestCase):
    def test_trivial(self):
        """ Test mean() on a stream of zeroes """
        stream = repeat(np.zeros((64, 64), dtype=np.float), times=5)
        for av in mean(stream):
            self.assertTrue(np.allclose(av, np.zeros_like(av)))

    def test_vs_numpy(self):
        """ Test mean vs. numpy.mean """
        stream = [np.random.random(size=(64, 64)) for _ in range(5)]
        stack = np.dstack(stream)

        for axis in (0, 1, 2, None):
            with self.subTest(f"axis = {axis}"):
                from_stream = mean(stream, axis=axis)
                from_numpy = np.mean(stack, axis=axis)
                self.assertTrue(np.allclose(from_numpy, from_stream))

    def test_against_numpy_nanmean(self):
        """ Test results against numpy.mean"""
        source = [np.random.random((16, 12, 5)) for _ in range(10)]
        for arr in source:
            arr[randint(0, 15), randint(0, 11), randint(0, 4)] = np.nan
        stack = np.stack(source, axis=-1)
        for axis in (0, 1, 2, None):
            with self.subTest(f"axis = {axis}"):
                from_numpy = np.nanmean(stack, axis=axis)
                out = mean(source, axis=axis, ignore_nan=True)
                self.assertSequenceEqual(from_numpy.shape, out.shape)
                self.assertTrue(np.allclose(out, from_numpy))


class TestIMean(unittest.TestCase):
    def test_against_numpy_mean(self):
        """ Test results against numpy.mean"""
        source = [np.random.random((16, 12, 5)) for _ in range(10)]
        stack = np.stack(source, axis=-1)
        for axis in (0, 1, 2, None):
            with self.subTest(f"axis = {axis}"):
                from_numpy = np.mean(stack, axis=axis)
                out = last(imean(source, axis=axis))
                self.assertSequenceEqual(from_numpy.shape, out.shape)
                self.assertTrue(np.allclose(out, from_numpy))

    def test_against_numpy_nanmean(self):
        """ Test results against numpy.mean"""
        source = [np.random.random((16, 12, 5)) for _ in range(10)]
        for arr in source:
            arr[randint(0, 15), randint(0, 11), randint(0, 4)] = np.nan
        stack = np.stack(source, axis=-1)
        for axis in (0, 1, 2, None):
            with self.subTest(f"axis = {axis}"):
                from_numpy = np.nanmean(stack, axis=axis)
                out = last(imean(source, axis=axis, ignore_nan=True))
                self.assertSequenceEqual(from_numpy.shape, out.shape)
                self.assertTrue(np.allclose(out, from_numpy))


class Testvar(unittest.TestCase):
    def test_vs_numpy(self):
        """ Test that the axis parameter is handled correctly """
        stream = [np.random.random((16, 7, 3)) for _ in range(5)]
        stack = np.stack(stream, axis=-1)

        for axis in (0, 1, 2, None):
            with self.subTest(f"axis = {axis}"):
                from_numpy = np.var(stack, axis=axis)
                from_var = var(stream, axis=axis)
                self.assertSequenceEqual(from_numpy.shape, from_var.shape)
                self.assertTrue(np.allclose(from_var, from_numpy))

    def test_ddof(self):
        """ Test that the ddof parameter is equivalent to numpy's """
        stream = [np.random.random((16, 7, 3)) for _ in range(10)]
        stack = np.stack(stream, axis=-1)

        with catch_warnings():
            simplefilter("ignore")
            for axis in (0, 1, 2, None):
                for ddof in range(4):
                    with self.subTest(f"axis = {axis}, ddof = {ddof}"):
                        from_numpy = np.var(stack, axis=axis, ddof=ddof)
                        from_var = var(stream, axis=axis, ddof=ddof)
                        self.assertSequenceEqual(from_numpy.shape, from_var.shape)
                        self.assertTrue(np.allclose(from_var, from_numpy))


class TestIvar(unittest.TestCase):
    def test_first(self):
        """ Test that the first yielded value of ivar is an array fo zeros """
        stream = repeat(np.random.random(size=(64, 64)), times=5)
        first = next(ivar(stream))

        self.assertTrue(np.allclose(first, np.zeros_like(first)))

    def test_output_shape(self):
        """ Test that the axis parameter is handled correctly """
        stream = [np.random.random((16, 7, 3)) for _ in range(5)]
        stack = np.stack(stream, axis=-1)

        for axis in (0, 1, 2, None):
            with self.subTest(f"axis = {axis}"):
                from_numpy = np.var(stack, axis=axis)
                from_ivar = last(ivar(stream, axis=axis))
                self.assertSequenceEqual(from_numpy.shape, from_ivar.shape)
                self.assertTrue(np.allclose(from_ivar, from_numpy))

    def test_ddof(self):
        """ Test that the ddof parameter is equivalent to numpy's """
        stream = [np.random.random((16, 7, 3)) for _ in range(10)]
        stack = np.stack(stream, axis=-1)

        with catch_warnings():
            simplefilter("ignore")
            for axis in (0, 1, 2, None):
                for ddof in range(4):
                    with self.subTest(f"axis = {axis}, ddof = {ddof}"):
                        from_numpy = np.var(stack, axis=axis, ddof=ddof)
                        from_ivar = last(ivar(stream, axis=axis, ddof=ddof))
                        self.assertSequenceEqual(from_numpy.shape, from_ivar.shape)
                        self.assertTrue(np.allclose(from_ivar, from_numpy))


class TestStd(unittest.TestCase):
    def test_against_numpy_std(self):
        stream = [np.random.random((16, 7, 3)) for _ in range(10)]
        stack = np.stack(stream, axis=-1)

        with catch_warnings():
            simplefilter("ignore")
            for axis in (0, 1, 2, None):
                for ddof in range(4):
                    with self.subTest(f"axis = {axis}, ddof = {ddof}"):
                        from_numpy = np.std(stack, axis=axis, ddof=ddof)
                        from_ivar = std(stream, axis=axis, ddof=ddof)
                        self.assertSequenceEqual(from_numpy.shape, from_ivar.shape)
                        self.assertTrue(np.allclose(from_ivar, from_numpy))

    def test_against_numpy_nanstd(self):
        source = [np.random.random((16, 12, 5)) for _ in range(10)]
        for arr in source:
            arr[randint(0, 15), randint(0, 11), randint(0, 4)] = np.nan
        stack = np.stack(source, axis=-1)

        for axis in (0, 1, 2, None):
            for ddof in range(4):
                with self.subTest(f"axis = {axis}, ddof = {ddof}"):
                    from_numpy = np.nanstd(stack, axis=axis, ddof=ddof)
                    from_ivar = std(source, axis=axis, ddof=ddof, ignore_nan=True)
                    self.assertSequenceEqual(from_numpy.shape, from_ivar.shape)
                    self.assertTrue(np.allclose(from_ivar, from_numpy))


class TestIStd(unittest.TestCase):
    def test_against_numpy_std(self):
        stream = [np.random.random((16, 7, 3)) for _ in range(10)]
        stack = np.stack(stream, axis=-1)

        with catch_warnings():
            simplefilter("ignore")
            for axis in (0, 1, 2, None):
                for ddof in range(4):
                    with self.subTest(f"axis = {axis}, ddof = {ddof}"):
                        from_numpy = np.std(stack, axis=axis, ddof=ddof)
                        from_ivar = last(istd(stream, axis=axis, ddof=ddof))
                        self.assertSequenceEqual(from_numpy.shape, from_ivar.shape)
                        self.assertTrue(np.allclose(from_ivar, from_numpy))

    def test_against_numpy_nanstd(self):
        source = [np.random.random((16, 12, 5)) for _ in range(10)]
        for arr in source:
            arr[randint(0, 15), randint(0, 11), randint(0, 4)] = np.nan
        stack = np.stack(source, axis=-1)

        for axis in (0, 1, 2, None):
            for ddof in range(4):
                with self.subTest(f"axis = {axis}, ddof = {ddof}"):
                    from_numpy = np.nanstd(stack, axis=axis, ddof=ddof)
                    from_ivar = last(
                        istd(source, axis=axis, ddof=ddof, ignore_nan=True)
                    )
                    self.assertSequenceEqual(from_numpy.shape, from_ivar.shape)
                    self.assertTrue(np.allclose(from_ivar, from_numpy))


@unittest.skipIf(not WITH_SCIPY, "SciPy is not installed/importable")
class TestSem(unittest.TestCase):
    def test_against_scipy_no_nans(self):
        """ Test that isem outputs the same as scipy.stats.sem """
        source = [np.random.random((16, 12, 5)) for _ in range(10)]
        stack = np.stack(source, axis=-1)

        for axis in (0, 1, 2, None):
            for ddof in range(4):
                with self.subTest(f"axis = {axis}, ddof = {ddof}"):
                    from_scipy = scipy_sem(stack, axis=axis, ddof=ddof)
                    from_isem = sem(source, axis=axis, ddof=ddof)
                    self.assertSequenceEqual(from_scipy.shape, from_isem.shape)
                    self.assertTrue(np.allclose(from_isem, from_scipy))

    def test_against_scipy_with_nans(self):
        """ Test that isem outputs the same as scipy.stats.sem when NaNs are ignored. """
        source = [np.random.random((16, 12, 5)) for _ in range(10)]
        for arr in source:
            arr[randint(0, 15), randint(0, 11), randint(0, 4)] = np.nan
        stack = np.stack(source, axis=-1)

        for axis in (0, 1, 2, None):
            for ddof in range(4):
                with self.subTest(f"axis = {axis}, ddof = {ddof}"):
                    from_scipy = scipy_sem(
                        stack, axis=axis, ddof=ddof, nan_policy="omit"
                    )
                    from_isem = sem(source, axis=axis, ddof=ddof, ignore_nan=True)
                    self.assertSequenceEqual(from_scipy.shape, from_isem.shape)
                    self.assertTrue(np.allclose(from_isem, from_scipy))


@unittest.skipIf(not WITH_SCIPY, "SciPy is not installed/importable")
class TestISem(unittest.TestCase):
    def test_against_scipy_no_nans(self):
        """ Test that isem outputs the same as scipy.stats.sem """
        source = [np.random.random((16, 12, 5)) for _ in range(10)]
        stack = np.stack(source, axis=-1)

        for axis in (0, 1, 2, None):
            for ddof in range(4):
                with self.subTest(f"axis = {axis}, ddof = {ddof}"):
                    from_scipy = scipy_sem(stack, axis=axis, ddof=ddof)
                    from_isem = last(isem(source, axis=axis, ddof=ddof))
                    self.assertSequenceEqual(from_scipy.shape, from_isem.shape)
                    self.assertTrue(np.allclose(from_isem, from_scipy))

    def test_against_scipy_with_nans(self):
        """ Test that isem outputs the same as scipy.stats.sem when NaNs are ignored. """
        source = [np.random.random((16, 12, 5)) for _ in range(10)]
        for arr in source:
            arr[randint(0, 15), randint(0, 11), randint(0, 4)] = np.nan
        stack = np.stack(source, axis=-1)

        for axis in (0, 1, 2, None):
            for ddof in range(4):
                with self.subTest(f"axis = {axis}, ddof = {ddof}"):
                    from_scipy = scipy_sem(
                        stack, axis=axis, ddof=ddof, nan_policy="omit"
                    )
                    from_isem = last(
                        isem(source, axis=axis, ddof=ddof, ignore_nan=True)
                    )
                    self.assertSequenceEqual(from_scipy.shape, from_isem.shape)
                    self.assertTrue(np.allclose(from_isem, from_scipy))


class TestIHistogram(unittest.TestCase):
    def test_against_numpy_no_weights(self):
        """ Test ihistogram against numpy.histogram with no weights """
        source = [np.random.random((16, 12, 5)) for _ in range(10)]
        stack = np.stack(source, axis=-1)

        bins = np.linspace(0, 1, num=10)
        from_numpy = np.histogram(stack, bins=bins)[0]
        from_ihistogram = last(ihistogram(source, bins=bins))

        # Since histogram output is int, cannot use allclose
        self.assertTrue(np.all(np.equal(from_numpy, from_ihistogram)))

    def test_trivial_weights(self):
        """ Test ihistogram with weights being all 1s vs. weights=None """
        source = [np.random.random((16, 12, 5)) for _ in range(10)]
        weights = [np.array([1]) for _ in source]

        bins = np.linspace(0, 1, num=10)
        none_weights = last(ihistogram(source, bins=bins, weights=None))
        trivial_weights = last(ihistogram(source, bins=bins, weights=weights))

        self.assertTrue(np.all(np.equal(none_weights, trivial_weights)))


if __name__ == "__main__":
    unittest.main()
