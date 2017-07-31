# -*- coding: utf-8 -*-
import unittest
from itertools import repeat
from random import randint, random

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
		stream = [np.random.random(size = (2,2)) for _ in range(5)]

		with self.subTest('float weights'):
			weights = [random() for _ in stream]
			from_iaverage = last(iaverage(stream, weights = weights))
			from_numpy = np.average(np.dstack(stream), axis = 2, weights = np.array(weights))

			self.assertTrue(np.allclose(from_iaverage, from_numpy))
		
		with self.subTest('array weights'):
			weights = [np.random.random(size = stream[0].shape) for _ in stream]
			from_iaverage = last(iaverage(stream, weights = weights))
			from_numpy= np.average(np.dstack(stream), axis = 2, weights = np.dstack(weights))

			self.assertTrue(np.allclose(from_iaverage, from_numpy))

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

if __name__ == '__main__':
	unittest.main()
