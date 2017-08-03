# -*- coding: utf-8 -*-
import unittest
from itertools import repeat
from random import randint, random
from warnings import catch_warnings, simplefilter

import numpy as np
from scipy.stats import sem as scipy_sem

from .. import iaverage, imean, isem, istd, ivar, last

class TestIAverage(unittest.TestCase):

    def test_trivial(self):
        """ Test iaverage on stream of zeroes """
        stream = repeat(np.zeros( (64,64), dtype = np.float ), times = 5)
        for av in iaverage(stream):
            self.assertTrue(np.allclose(av, np.zeros_like(av)))
    
    def test_average(self):
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
    
    @unittest.skip('')
    def test_ignore_nan(self):
        """ Test that NaNs are handled correctly """
        stream = [np.random.random(size = (16,16)) for _ in range(5)]
        for s in stream:
            s[randint(0, 15), randint(0,15)] = np.nan
        
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

    def test_against_numpy(self):
        """ Test results against numpy.average """
        source = [np.zeros((16, 12)) for _ in range(10)]
        stack = np.stack(source, axis = -1)
        for axis in (0, 1, None):
            with self.subTest('axis = {}'.format(axis)):
                from_numpy = np.average(stack, axis = axis)
                out = last(iaverage(source, axis = axis))
                self.assertTrue(np.allclose(out, from_numpy))

class TestIMean(unittest.TestCase):

    def test_trivial(self):
        """ Test iaverage on stream of zeroes """
        stream = repeat(np.zeros( (64,64), dtype = np.float ), times = 5)
        for av in imean(stream):
            self.assertTrue(np.allclose(av, np.zeros_like(av)))
    
    def test_mean(self):
        """ Test results against of unweighted average against numpy.mean """
        stream = [np.random.random(size = (64,64)) for _ in range(5)]

        from_imean = last(imean(stream))
        from_numpy = np.mean(np.dstack(stream), axis = 2)

        self.assertTrue(np.allclose(from_imean, from_numpy))

class TestISem(unittest.TestCase):

    def test_first(self):
        """ Test that the first yielded value of isem is an array fo zeros """
        stream = repeat(np.random.random( size = (64,64)), times = 5)
        first = next(isem(stream))

        self.assertTrue(np.allclose(first, np.zeros_like(first)))
    
    def test_against_scipy_sem(self):
        """ Test that the results of isem are in agreement with scipy.stats.sem """
        stream = [np.random.random(size = (64,64)) for _ in range(5)]

        for ddof in range(0, len(stream)):
            with self.subTest('ddof = {}'.format(ddof)):
                from_isem = last(isem(stream, ddof = ddof))
                from_scipy = scipy_sem(np.dstack(stream), axis = 2, ddof = ddof)

                self.assertTrue(np.allclose(from_isem, from_scipy))

class TestIstd(unittest.TestCase):

    def test_first(self):
        """ Test that the first yielded value of istd is an array fo zeros """
        stream = repeat(np.random.random( size = (64,64)), times = 5)
        first = next(istd(stream))

        self.assertTrue(np.allclose(first, np.zeros_like(first)))

    def test_against_numpy_std(self):
        """ Test that the results of istd are in agreement with numpy.std """
        stream = [np.random.random(size = (64,64)) for _ in range(5)]

        for ddof in range(0, len(stream)):
            with self.subTest('ddof = {}'.format(ddof)):
                from_istd = last(istd(stream, ddof = ddof))
                from_numpy = np.std(np.dstack(stream), axis = 2, ddof = ddof)

                self.assertTrue(np.allclose(from_istd, from_numpy))

    def test_weighted_std(self):
        """ Test that weighted streaming std gives correct results """
        stream = [np.random.random(size = (64,64)) for _ in range(5)]

        with self.subTest('float weights'):
            weights = [random() for _ in stream]
            from_istd = last(istd(stream, ddof = 0, weights = weights))
            
            # Numpy/scipy does not have a weighted variance function at this time
            arr = np.dstack(stream)
            average = np.average(arr, weights = weights, axis = 2)
            wvar = np.average((arr - average[:,:,None])**2, weights = weights, axis = 2) 	# weighted variance

            self.assertTrue(np.allclose(from_istd, np.sqrt(wvar)))

        with self.subTest('array weights'):
            weights = [np.random.random(size = stream[0].shape) for _ in stream]
            from_istd = last(istd(stream, ddof = 0, weights = weights))
            
            # Numpy/scipy does not have a weighted variance function at this time
            arr = np.dstack(stream)
            weights = np.dstack(weights)
            average = np.average(arr, weights = weights, axis = 2)
            wvar = np.average((arr - average[:,:,None])**2, weights = weights, axis = 2) 	# weighted variance

            self.assertTrue(np.allclose(from_istd, np.sqrt(wvar)))

    def test_ignore_nan(self):
        """ Test that NaNs are handled correctly """
        stream = [np.random.random(size = (16,16)) for _ in range(5)]
        for s in stream:
            s[randint(0, 15), randint(0,15)] = np.nan
        
        with catch_warnings():
            simplefilter('ignore')
            from_istd = last(istd(stream, ignore_nan = True))  
        from_numpy = np.nanstd(np.dstack(stream), axis = 2)
        self.assertTrue(np.allclose(from_istd, from_numpy))

class TestIvar(unittest.TestCase):

    def test_first(self):
        """ Test that the first yielded value of ivar is an array fo zeros """
        stream = repeat(np.random.random( size = (64,64)), times = 5)
        first = next(ivar(stream))

        self.assertTrue(np.allclose(first, np.zeros_like(first)))

    def test_against_numpy_var(self):
        """ Test that the results of istd are in agreement with numpy.var """
        stream = [np.random.random(size = (64,64)) for _ in range(5)]

        for ddof in range(0, len(stream)):
            with self.subTest('ddof = {}'.format(ddof)):
                from_ivar = last(ivar(stream, ddof = ddof))
                from_numpy = np.var(np.dstack(stream), axis = 2, ddof = ddof)

                self.assertTrue(np.allclose(from_ivar, from_numpy))

    def test_weighted_variance(self):
        """ Test that weighted streaming variance gives correct results """
        stream = [np.random.random(size = (64,64)) for _ in range(5)]

        with self.subTest('float weights'):
            weights = [random() for _ in stream]
            from_ivar = last(ivar(stream, ddof = 0, weights = weights))
            
            # Numpy/scipy does not have a weighted variance function at this time
            arr = np.dstack(stream)
            average = np.average(arr, weights = weights, axis = 2)
            weighted = np.average((arr - average[:,:,None])**2, weights = weights, axis = 2) 

            self.assertTrue(np.allclose(from_ivar, weighted))

        with self.subTest('array weights'):
            weights = [np.random.random(size = stream[0].shape) for _ in stream]
            from_ivar = last(ivar(stream, ddof = 0, weights = weights))
            
            # Numpy/scipy does not have a weighted variance function at this time
            arr = np.dstack(stream)
            weights = np.dstack(weights)
            average = np.average(arr, weights = weights, axis = 2)
            weighted = np.average((arr - average[:,:,None])**2, weights = weights, axis = 2) 

            self.assertTrue(np.allclose(from_ivar, weighted))
        
    def test_ignore_nan(self):
        """ Test that NaNs are handled correctly """
        stream = [np.random.random(size = (16,16)) for _ in range(5)]
        for s in stream:
            s[randint(0, 15), randint(0,15)] = np.nan
        
        with catch_warnings():
            simplefilter('ignore')
            from_ivar = last(ivar(stream, ignore_nan = True))  
        from_numpy = np.nanvar(np.dstack(stream), axis = 2)
        self.assertTrue(np.allclose(from_ivar, from_numpy))

    def test_axis(self):
        """ Test that the axis parameter is handled correctly """
        stream = [np.zeros((16,)) for _ in range(5)]

        with self.subTest('axis = 0'):
            var = last(ivar(stream, axis = 0))
            self.assertEqual(var, 0)
        
        with self.subTest('axis = None'):
            var = last(ivar(stream, axis = None))
            self.assertEqual(var, 0)

if __name__ == '__main__':
    unittest.main()
