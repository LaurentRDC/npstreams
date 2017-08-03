# -*- coding: utf-8 -*-
import unittest
from itertools import repeat
from random import randint, random, seed
from warnings import catch_warnings, simplefilter

import numpy as np
from scipy.stats import sem as scipy_sem

from .. import iaverage, imean, isem, istd, ivar, last

seed(23)

class TestIAverage(unittest.TestCase):

    def test_trivial(self):
        """ Test iaverage on stream of zeroes """
        stream = repeat(np.zeros( (64,64), dtype = np.float ), times = 5)
        for av in iaverage(stream):
            self.assertTrue(np.allclose(av, np.zeros_like(av)))
    
    def test_weighted_average(self):
        """ Test results of weighted average against numpy.average """
        stream = [np.random.random(size = (16,16)) for _ in range(5)]

        with self.subTest('float weights'):
            weights = [random() for _ in stream]
            from_iaverage = last(iaverage(stream, weights = weights))
            from_numpy = np.average(np.dstack(stream), axis = 2, weights = np.array(weights))
            self.assertTrue(np.allclose(from_iaverage, from_numpy))
        
        with self.subTest('array weights'):
            weights = [np.random.random(size = stream[0].shape) for _ in stream]
            from_iaverage = last(iaverage(stream, weights = weights))
            from_numpy = np.average(np.dstack(stream), axis = 2, weights = np.dstack(weights))
            self.assertTrue(np.allclose(from_iaverage, from_numpy))
    
    def test_ignore_nan(self):
        """ Test that NaNs are handled correctly """
        stream = [np.random.random(size = (16,12)) for _ in range(5)]
        for s in stream:
            s[randint(0, 15), randint(0,11)] = np.nan
        
        with catch_warnings():
            simplefilter('ignore')
            from_iaverage = last(iaverage(stream, ignore_nan = True))  
        from_numpy = np.nanmean(np.dstack(stream), axis = 2)
        self.assertTrue(np.allclose(from_iaverage, from_numpy))
    
    def test_length(self):
        """ Test that the number of yielded elements is the same as source """
        source = (np.zeros((16,)) for _ in range(5))
        avg = list(iaverage(source, axis = 0))
        self.assertEqual(len(avg), 5)
    
    def test_output_shape(self):
        """ Test output shape """
        source = [np.random.random((16, 12, 5)) for _ in range(10)]
        stack = np.stack(source, axis = -1)
        for axis in (0, 1, 2, None):
            with self.subTest('axis = {}'.format(axis)):
                from_numpy = np.average(stack, axis = axis)
                out = last(iaverage(source, axis = axis))
                self.assertSequenceEqual(from_numpy.shape, out.shape)
                self.assertTrue(np.allclose(out, from_numpy))

class TestIMean(unittest.TestCase):
    
    def test_against_numpy_mean(self):
        """ Test results against numpy.mean"""
        source = [np.random.random((16, 12, 5)) for _ in range(10)]
        stack = np.stack(source, axis = -1)
        for axis in (0, 1, 2, None):
            with self.subTest('axis = {}'.format(axis)):
                from_numpy = np.mean(stack, axis = axis)
                out = last(imean(source, axis = axis))
                self.assertSequenceEqual(from_numpy.shape, out.shape)
                self.assertTrue(np.allclose(out, from_numpy))

    def test_against_numpy_nanmean(self):
        """ Test results against numpy.mean"""
        source = [np.random.random((16, 12, 5)) for _ in range(10)]
        for arr in source:
            arr[randint(0, 15), randint(0, 11), randint(0, 4)] = np.nan
        stack = np.stack(source, axis = -1)
        for axis in (0, 1, 2, None):
            with self.subTest('axis = {}'.format(axis)):
                from_numpy = np.nanmean(stack, axis = axis)
                out = last(imean(source, axis = axis, ignore_nan = True))
                self.assertSequenceEqual(from_numpy.shape, out.shape)
                self.assertTrue(np.allclose(out, from_numpy))

class TestIvar(unittest.TestCase):

    def test_first(self):
        """ Test that the first yielded value of ivar is an array fo zeros """
        stream = repeat(np.random.random( size = (64,64)), times = 5)
        first = next(ivar(stream))

        self.assertTrue(np.allclose(first, np.zeros_like(first)))

    def test_output_shape(self):
        """ Test that the axis parameter is handled correctly """
        stream = [np.random.random((16, 7, 3)) for _ in range(5)]
        stack = np.stack(stream, axis = -1)

        for axis in (0, 1, 2, None):
            with self.subTest('axis = {}'.format(axis)):
                from_numpy = np.var(stack, axis = axis)
                from_ivar = last(ivar(stream, axis = axis))
                self.assertSequenceEqual(from_numpy.shape, from_ivar.shape)
                self.assertTrue(np.allclose(from_ivar, from_numpy))

    def test_ddof(self):
        """ Test that the ddof parameter is equivalent to numpy's """
        stream = [np.random.random((16, 7, 3)) for _ in range(10)]
        stack = np.stack(stream, axis = -1)

        for axis in (0, 1, 2, None):
            for ddof in range(4):
                with self.subTest('axis = {}, ddof = {}'.format(axis, ddof)):
                    from_numpy = np.var(stack, axis = axis, ddof = ddof)
                    from_ivar = last(ivar(stream, axis = axis, ddof = ddof))
                    self.assertSequenceEqual(from_numpy.shape, from_ivar.shape)
                    self.assertTrue(np.allclose(from_ivar, from_numpy))

class TestIStd(unittest.TestCase):

    def test_against_numpy_std(self):
        stream = [np.random.random((16, 7, 3)) for _ in range(10)]
        stack = np.stack(stream, axis = -1)

        for axis in (0, 1, 2, None):
            for ddof in range(4):
                with self.subTest('axis = {}, ddof = {}'.format(axis, ddof)):
                    from_numpy = np.std(stack, axis = axis, ddof = ddof)
                    from_ivar = last(istd(stream, axis = axis, ddof = ddof))
                    self.assertSequenceEqual(from_numpy.shape, from_ivar.shape)
                    self.assertTrue(np.allclose(from_ivar, from_numpy))

    def test_against_numpy_nanstd(self):
        source = [np.random.random((16, 12, 5)) for _ in range(10)]
        for arr in source:
            arr[randint(0, 15), randint(0, 11), randint(0, 4)] = np.nan
        stack = np.stack(source, axis = -1)

        for axis in (0, 1, 2, None):
            for ddof in range(4):
                with self.subTest('axis = {}, ddof = {}'.format(axis, ddof)):
                    from_numpy = np.nanstd(stack, axis = axis, ddof = ddof)
                    from_ivar = last(istd(source, axis = axis, ddof = ddof, ignore_nan = True))
                    self.assertSequenceEqual(from_numpy.shape, from_ivar.shape)
                    self.assertTrue(np.allclose(from_ivar, from_numpy))

if __name__ == '__main__':
    unittest.main()
