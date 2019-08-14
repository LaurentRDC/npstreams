.. include:: references.txt

.. _recipes:

*******
Recipes
*******

Single-pass mean and error calculation
--------------------------------------

Here is a snipped for a function that computes a mean
and standard error in the mean (SEM) in a single pass::

    from npstreams import imean, isem, array_stream, itercopy
    
    # The `array_stream` decorator ensures that the elements of 
    # the iterable `arrays` will be converted to ndarrays if possible
    # This decorator is not required.
    @array_stream   
    def mean_and_error(arrays, axis = -1):
        """ Yields (mean, error) pairs from a stream of arrays """
        # itercopy creates a copy of the original stream
        # The elements are only generated once, and then fed
        # to those two copies; much more efficient than
        # creating two streams from scratch.
        arrays_for_mean, arrays_for_sem = itercopy(arrays)

        means = imean(arrays_for_mean, axis = axis)
        errors = isem(arrays_for_sem, axis = axis)

        yield from zip(means, errors)