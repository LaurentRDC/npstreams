npstreams
=========

.. image:: https://img.shields.io/appveyor/ci/LaurentRDC/npstreams/master.svg
    :target: https://ci.appveyor.com/project/LaurentRDC/npstreams
    :alt: Windows Build Status
.. image:: https://readthedocs.org/projects/npstreams/badge/?version=latest
    :target: http://npstreams.readthedocs.io
    :alt: Documentation Build Status
.. image:: https://img.shields.io/pypi/v/npstreams.svg
    :target: https://pypi.python.org/pypi/npstreams
    :alt: PyPI Version

npstreams is an open-source Python package for streaming NumPy array operations. 
The goal is to provide tested routines that operate on streams (or generators) of arrays instead of dense arrays.

Streaming reduction operations (sums, averages, etc.) can be implemented in constant memory, which in turns
allows for easy parallelization. Some routines in npstreams are parallelized in this way.

In my experience, this approach has been a godsend when working with images; the images are read
one-by-one from disk and combined/processed in a streaming fashion.

Motivating Example
------------------

Consider the following snippet to combine 50 images 
from an iterable :code:`source`::

	import numpy as np

	images = np.empty( shape = (2048, 2048, 50) )
	from index, im in enumerate(source):
	    images[:,:,index] = im
	
	avg = np.average(images, axis = 2)

If the :code:`source` iterable provided 1000 images, the above routine would
not work on most machines. Moreover, what if we want to transform the images 
one by one before averaging them? What about looking at the average while it 
is being computed? Let's look at an example::

	import numpy as np
	from npstreams import iaverage
	from scipy.misc import imread

	stream = map(imread, list_of_filenames)
	averaged = iaverage(stream)

At this point, the generators :code:`map` and :code:`iaverage` are 'wired'
but will not compute anything until it is requested. We can look at the average evolve::

    import matplotlib.pyplot as plt
    for avg in average:
        plt.imshow(avg); plt.show()

We can also use :code:`last` to get at the final average::

	from npstreams import last

	total = last(averaged) # average of the entire stream

Streaming Functions
-------------------

npstreams comes with some streaming functions built-in. Some examples:

* Numerics : :code:`isum`, :code:`iprod`, :code:`isub`, etc.
* Statistics : :code:`iaverage` (weighted mean), :code:`ivar` (single-pass variance), etc.
* Stacking : :code:`iflatten`, :code:`istack`

All routines are documented in the `API Reference on readthedocs.io <http://npstreams.readthedocs.io>`_.

Making your own Streaming Functions
-----------------------------------

Any NumPy reduction function can be transformed into a streaming function using the
:code:`stream_reduce` function. For example::

    from npstreams import stream_reduce
    from numpy import prod

    def streaming_prod(stream, axis, **kwargs):
        """ Streaming product along axis """
        yield from stream_reduce(stream, npfunc = prod, axis = axis, **kwargs)

The above :code:`streaming_prod` will accumulate (and yield) the result of the operation
as arrays come in the stream. 

The two following snippets should return the same result::

    from numpy import prod, stack
    
    dense = stack(stream, axis = -1) 
    from_numpy = prod(dense, axis = 0)  # numpy.prod = numpy.multiply.reduce

.. code::

    from npstreams import last

    from_stream = last(streaming_prod(stream, axis = 0))

However, :code:`streaming_prod` will work on 100 GB of data in a single line of code.

Future Work
-----------
Some of the features I want to implement in this package in the near future:

* Benchmark section : how does the performance compare with NumPy functions, as array size increases?
* Cython : cythonizing the underlying routines would probably help.
* More functions : more streaming functions borrowed from NumPy and SciPy.

API Reference
-------------

The `API Reference on readthedocs.io <http://npstreams.readthedocs.io>`_ provides API-level documentation, as 
well as tutorials.

Installation
------------

scikit-ued is available on PyPI; it can be installed with `pip <https://pip.pypa.io>`_.::

    python -m pip install npstreams

To install the latest development version from `Github <https://github.com/LaurentRDC/npstreams>`_::

    python -m pip install git+git://github.com/LaurentRDC/npstreams.git

Each version is tested against Python 3.4, 3.5 and 3.6. If you are using a different version, tests can be run
using the standard library's `unittest` module.

Support / Report Issues
-----------------------

All support requests and issue reports should be
`filed on Github as an issue <https://github.com/LaurentRDC/npstreams/issues>`_.

License
-------

npstreams is made available under the BSD License, same as NumPy. For more details, see `LICENSE.txt <https://github.com/LaurentRDC/npstreams/blob/master/LICENSE.txt>`_.
