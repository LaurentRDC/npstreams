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

        stream = [0, 1, np.array([1])]
        for arr in iden(stream):
            self.assertIsInstance(arr, np.ndarray)
        
    def test_single_array(self):
        """ Test that a 'stream' consisting of a single array is repackaged into an iterable """
        stream = np.array([1,2,3])
        self.assertEqual(len(list(iden(stream))), 1)

class TestArrayStream(unittest.TestCase):

    def test_length_hint_sized_iterable(self):
        """ Test the accuracy of __length_hint__ for ArrayStream constructed
        from a sized iterable """
        iterable = [1,2,3,4,5]
        a = ArrayStream(iterable)
        self.assertEqual(len(iterable), a.__length_hint__())

    def test_length_hint_not_sized_iterable(self):
        """ Test that __length_hint__ returns NotImplemented for ArrayStream constructed
        from an unsized iterable """
        iterable = (0 for _ in range(10))
        a = ArrayStream(iterable)
        self.assertIs(a.__length_hint__(), NotImplemented)
    
    def test_conversion_to_array(self):
        """ Test that numpy.array(Arraystream(...)) returns an array built as a stack of arrays """
        a = ArrayStream([np.random.random((16,16)) for _ in range(10)])
        arr = np.array(a)
        self.assertEqual(arr.shape, (16,16,10))


if __name__ == '__main__':
	unittest.main()
