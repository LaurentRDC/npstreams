npstreams
=========

.. image:: https://img.shields.io/appveyor/ci/LaurentRDC/npstreams/master.svg
    :target: https://ci.appveyor.com/project/LaurentRDC/npstreams
    :alt: Windows Build Status
.. image:: https://readthedocs.org/projects/npstreams/badge/?version=master
    :target: http://npstreams.readthedocs.io
    :alt: Documentation Build Status
.. image:: https://img.shields.io/pypi/v/npstreams.svg
    :target: https://pypi.python.org/pypi/npstreams
    :alt: PyPI Version
.. image:: https://img.shields.io/conda/vn/conda-forge/npstreams.svg
    :target: https://anaconda.org/conda-forge/npstreams
    :alt: Conda-forge Version
.. image:: https://img.shields.io/pypi/pyversions/npstreams.svg
    :alt: Supported Python Versions
.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :alt: Code formatting style
    :target: https://github.com/ambv/black

npstreams is an open-source Python package for streaming NumPy array operations. 
The goal is to provide tested routines that operate on streams (or generators) of arrays instead of dense arrays.
Some routines are CUDA-enabled, based on `PyCUDA <https://documen.tician.de/pycuda/>`_'s GPUArray (work-in-progress).

Streaming reduction operations (sums, averages, etc.) can be implemented in constant memory, which in turns
allows for easy parallelization.

This approach has been a huge boon when working with lots of images; the images are read
one-by-one from disk and combined/processed in a streaming fashion.

This package is developed in conjunction with other software projects in the 
`Siwick research group <http://www.physics.mcgill.ca/siwicklab/>`_.

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

More importantly, npstreams gives you all the tools required to build your own streaming function.
All routines are documented in the `API Reference on readthedocs.io <http://npstreams.readthedocs.io>`_.

Benchmarking
------------

npstreams provides a function for benchmarking common use cases.

To run the benchmark with default parameters, from the interpreter::

    from npstreams import benchmark
    benchmark()

From a command-line terminal::

    python -c 'import npstreams; npstreams.benchmark()'

The results will be printed to the screen.

Future Work
-----------
Some of the features I want to implement in this package in the near future:

* Optimize the CUDA-enabled routines
* More functions : more streaming functions borrowed from NumPy and SciPy.

API Reference
-------------

The `API Reference on readthedocs.io <http://npstreams.readthedocs.io>`_ provides API-level documentation, as 
well as tutorials.

Installation
------------

The only requirement is NumPy. To have access to CUDA-enabled routines, PyCUDA must also be
installed. npstreams is available on PyPI; it can be installed with `pip <https://pip.pypa.io>`_.::

    python -m pip install npstreams

npstreams can also be installed with the conda package manager, from the conda-forge channel::

    conda config --add channels conda-forge
    conda install npstreams

To install the latest development version from `Github <https://github.com/LaurentRDC/npstreams>`_::

    python -m pip install git+git://github.com/LaurentRDC/npstreams.git

Each version is tested against Python 3.6+. If you are using a different version, tests can be run
using the standard library's `unittest` module.

Citations
---------

If you find this software useful, please consider citing the following publication:

.. [#] L. P. René de Cotret, M. R. Otto, M. J. Stern. and B. J. Siwick, *An open-source software ecosystem for the interactive 
       exploration of ultrafast electron scattering data*, Advanced Structural and Chemical Imaging 4:11 (2018) DOI: 10.1186/s40679-018-0060-y

Support / Report Issues
-----------------------

All support requests and issue reports should be
`filed on Github as an issue <https://github.com/LaurentRDC/npstreams/issues>`_.

License
-------

npstreams is made available under the BSD License, same as NumPy. For more details, see `LICENSE.txt <https://github.com/LaurentRDC/npstreams/blob/master/LICENSE.txt>`_.
