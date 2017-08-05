.. include:: references.txt

.. _recipes:

*******
Recipes
*******

Single-pass mean and error calculation
--------------------------------------

Here is a snipped for a function that computes a mean
and standard error in the mean (SEM) in a single pass::

    from itertools import tee
    from npstreams import imean, isem

    def mean_and_error(arrays, axis = -1):
        """ Yields (mean, error) pairs from a stream of arrays """
        # itertools.tee splits a stream into two 'copies'
        # The elements are only generated once, and then fed
        # to those two copies; much more efficient than
        # creating two streams from scratch.
        arrays_for_mean, arrays_for_sem = tee(arrays)

        means = imean(arrays_for_mean, axis = axis)
        errors = isem(arrays_for_sem, axis = axis)

        yield from zip(means, errors)