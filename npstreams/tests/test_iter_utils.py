# -*- coding: utf-8 -*-
import unittest
from itertools import repeat
from .. import last, chunked, linspace, multilinspace, cyclic, length_hint


class TestLast(unittest.TestCase):
    def test_trivial(self):
        """ Test last() on iterable of identical values """
        i = repeat(1, 10)
        self.assertEqual(last(i), 1)

    def test_on_empty_iterable(self):
        """ Test that last() raises RuntimeError for empty iterable """
        with self.assertRaises(RuntimeError):
            last(list())


class TestCyclic(unittest.TestCase):
    def test_numbers(self):
        """ """
        permutations = set(cyclic((1, 2, 3)))
        self.assertIn((1, 2, 3), permutations)
        self.assertIn((2, 3, 1), permutations)
        self.assertIn((3, 1, 2), permutations)
        self.assertEqual(len(permutations), 3)


class TestLinspace(unittest.TestCase):
    def test_endpoint(self):
        """ Test that the endpoint is included by linspace() when appropriate"""
        with self.subTest("endpoint = True"):
            space = linspace(0, 1, num=10, endpoint=True)
            self.assertEqual(last(space), 1)

        with self.subTest("endpoint = False"):
            space = linspace(0, 1, num=10, endpoint=False)
            self.assertAlmostEqual(last(space), 0.9)

    def test_length(self):
        """ Test that linspace() returns an iterable of the correct length """
        with self.subTest("endpoint = True"):
            space = list(linspace(0, 1, num=13, endpoint=True))
            self.assertEqual(len(space), 13)

        with self.subTest("endpoint = False"):
            space = list(linspace(0, 1, num=13, endpoint=False))
            self.assertEqual(len(space), 13)


class TestMultilinspace(unittest.TestCase):
    def test_endpoint(self):
        """ Test that the endpoint is included by linspace() when appropriate"""
        with self.subTest("endpoint = True"):
            space = multilinspace((0, 0), (1, 1), num=10, endpoint=True)
            self.assertSequenceEqual(last(space), (1, 1))

        with self.subTest("endpoint = False"):
            space = multilinspace((0, 0), (1, 1), num=10, endpoint=False)
            # Unfortunately there is no assertSequenceAlmostEqual
            self.assertSequenceEqual(
                last(space), (0.8999999999999999, 0.8999999999999999)
            )

    def test_length(self):
        """ Test that linspace() returns an iterable of the correct length """
        with self.subTest("endpoint = True"):
            space = list(multilinspace((0, 0), (1, 1), num=13, endpoint=True))
            self.assertEqual(len(space), 13)

        with self.subTest("endpoint = False"):
            space = list(multilinspace((0, 0), (1, 1), num=13, endpoint=False))
            self.assertEqual(len(space), 13)


class TestChunked(unittest.TestCase):
    def test_larger_chunksize(self):
        """ Test chunked() with a chunksize larger that the iterable itself """
        i = repeat(1, 10)
        chunks = chunked(i, chunksize=15)
        self.assertEqual(len(list(chunks)), 1)  # One single chunk is returned

    def test_on_infinite_generator(self):
        """ Test chunked() on an infinite iterable """
        i = repeat(1)
        chunks = chunked(i, chunksize=15)
        for _ in range(10):
            self.assertEqual(len(next(chunks)), 15)

    def test_chunked_nonint_chunksize(self):
        """ Test that chunked raises a TypeError immediately if `chunksize` is not an integer """
        with self.assertRaises(TypeError):
            i = repeat(1)
            chunks = chunked(i, chunksize=15.0)


class TestLengthHint(unittest.TestCase):
    def test_on_sized(self):
        """ Test length_hint on a sized iterable """
        l = [1, 2, 3, 4, 5]
        self.assertEqual(length_hint(l), len(l))

    def test_on_unsized(self):
        """ Test length_hint on an unsized iterable returns the default """
        l = (0 for _ in range(10))
        self.assertEqual(length_hint(l, default=0), 0)

    def test_on_method_if_implemented(self):
        """ Test length_hint returns the same as __length_hint__ if implemented """

        class WithHint:
            """ Some dummy class with a length hint """

            def __length_hint__(self):
                return 1

        self.assertEqual(length_hint(WithHint(), default=0), 1)


if __name__ == "__main__":
    unittest.main()
