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
allows for easy parallelization.

In my experience, this approach has been a godsend when working with images; the images are read
one-by-one from disk and combined/processed in a streaming fashion.

This package is developed in conjunction with other software projects in the 
`Siwick research group <www.physics.mcgill.ca/siwicklab>`_

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

Example: Streaming Maximum
--------------------------

Let's create a streaming maximum function for a stream. First, we have to choose 
how to handle NaNs:

* If we want to propagate NaNs, we should use :code:`numpy.maximum`
* If we want to ignore NaNs, we should use :code:`numpy.fmax`

Both of those functions are binary ufuncs, so we can use :code:`ireduce_ufunc`. We will
also want to make sure that anything in the stream that isn't an array will be made into one
using the :code:`array_stream` decorator.

Putting it all together::

    from npstreams import array_stream, ireduce_ufunc
    from numpy import maximum, fmax

    @array_stream
    def imax(arrays, axis = -1, ignore_nan = False, **kwargs):
        """
        Streaming maximum along an axis.

        Parameters
        ----------
        arrays : iterable
            Stream of arrays to be compared.
        axis : int or None, optional
            Axis along which to compute the maximum. If None, 
            arrays are flattened before reduction.
        ignore_nan : bool, optional
            If True, NaNs are ignored. Default is False.
        
        Yields
        ------
        online_max : ndarray
        """
        ufunc = fmax if ignore_nan else maximum
        yield from ireduce_ufunc(arrays, ufunc, axis = axis, **kwargs)

This will provide us with a streaming function, meaning that we can look at the progress
as it is being computer. We can also create a function that returns the max of the stream
like :code:`numpy.ndarray.max()` using the :code:`npstreams.last` function::

    from npstreams import last

    def smax(*args, **kwargs):  # s for stream
        """
        Maximum of all arrays in a stream, along an axis.

        Parameters
        ----------
        arrays : iterable
            Stream of arrays to be compared.
        axis : int or None, optional
            Axis along which to compute the maximum. If None, 
            arrays are flattened before reduction.
        ignore_nan : bool, optional
            If True, NaNs are ignored. Default is False.
        
        Returns
        -------
        max : scalar or ndarray
        """
        return last(imax(*args, **kwargs)

Future Work
-----------
Some of the features I want to implement in this package in the near future:

* Benchmark section : how does the performance compare with NumPy functions, as array size increases?
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
