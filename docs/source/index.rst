.. include:: references.txt

.. _npstreams:

**************************************
`npstreams`: streaming NumPy functions
**************************************

:mod:`npstreams` is an open-source Python package for streaming NumPy array operations. 
The goal is to provide tested, (almost) drop-in replacements for NumPy functions (where possible) 
that operate on streams of arrays instead of dense arrays.

:mod:`npstreams` also provides some utilities for parallelization. These parallelization
generators can be combined with the streaming functions to drastically improve performance
in some cases.

The code presented herein has been in use at some point by the 
`Siwick research group <http://www.physics.mcgill.ca/siwicklab>`_.

.. warning::
        This code is in development and may break without warning.

Example
=======

Consider the following snippet to combine 50 images 
from an iterable :data:`source`::

	import numpy as np

	images = np.empty( shape = (2048, 2048, 50) )
	from index, im in enumerate(source):
	    images[:,:,index] = im
	
	avg = np.average(images, axis = 2)

If the :data:`source` iterable provided 10000 images, the above routine would
not work on most machines. Moreover, what if we want to transform the images 
one by one before averaging them? What about looking at the average while it 
is being computed? Let's look at an example::

	import numpy as np
	from npstreams import iaverage
	from scipy.misc import imread

	stream = map(imread, list_of_filenames)
	averaged = iaverage(stream)

At this point, the generators :func:`map` and :func:`iaverage` are 'wired'
but will not compute anything until it is requested. We can look at the average evolve::

    import matplotlib.pyplot as plt
    for avg in average:
        plt.imshow(avg); plt.show()

We can also use :func:`last` to get at the final average::

	from npstreams import last

	total = last(averaged) # average of the entire stream

Making your own streaming functions
===================================

Any **binary** NumPy Ufunc function can be transformed into a streaming function using the
:func:`ireduce_ufunc` function. For example::

    from npstreams import stream_reduce
    from numpy import prod

    def streaming_prod(stream, **kwargs):
        """ Streaming product along axis """
        yield from stream_ufuc(stream, ufunc = np.multiply, **kwargs)

The above :func:`streaming_prod` will accumulate (and yield) the result of the operation
as arrays come in the stream. 

The two following snippets should return the same result::

    from numpy import prod, stack
    
    dense = stack(stream, axis = -1) 
    from_numpy = prod(dense, axis = 0)  # numpy.prod = numpy.multiply.reduce

.. code::

    from npstreams import last

    from_stream = last(streaming_prod(stream, axis = 0))

However, :func:`streaming_prod` will work on 100 GB of data in a single line of code.

Benchmark
=========

Let's look at a simple benchmark. Let compare the two snippets to sum the following data::

    def stream():
        for _ in range(50):
            yield np.empty((2048, 2048), dtype = np.int)

Snippet 1: dense arrays only. Note that I count the creation of the dense array::

    import numpy as np

    stack = np.stack(list(stream()), axis = -1)
    s = np.sum(stack, axis = -1)

On my machine, this takes 4.93s and ~3G of memory.

Snippet 2: streaming arrays. This also includes the creation of the stream::

    # snippet 2
    import npstreams as nps
    s = nps.last(nps.isum(stream(), axis = -1))

On my machine, this takes 2.93s seconds and 95 MB of memory.

Bottom line: If your data is not already in memory, the routines of npstreams will be faster.

Links
=====

* `Source code <https://github.com/LaurentRDC/npstreams>`_
* `Issues <https://github.com/LaurentRDC/npstreams/issues>`_
* `Docs <http://npstreams.readthedocs.org/>`_

.. _npstreams_docs:

General Documentation
=====================

.. toctree::
    :maxdepth: 3
    
    installation
    conventions
    api
    cuda
    control_flow
    making_your_own
    recipes

Authors
=======

* Laurent P. René de Cotret