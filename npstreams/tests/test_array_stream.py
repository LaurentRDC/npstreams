# -*- coding: utf-8 -*-
import unittest
import numpy as np

from ..array_stream import array_stream, ArrayStream


@array_stream
def iden(arrays):
    yield from arrays

class TestArrayStreamDecorator(unittest.TestCase):

    def test_type(self):
        """ Test that all object from an array stream are ndarrays """

        stream = [0, 1, np.array([1]), 'a']
        for arr in iden(stream):
            self.assertIsInstance(arr, np.ndarray)
        
    def test_single_array(self):
        """ Test that a 'stream' consisting of a single array is repackaged into an iterable """
        stream = np.array([1,2,3])
        self.assertEqual(len(list(iden(stream))), 1)

if __name__ == '__main__':
	unittest.main()
