npstreams
=========

.. image:: https://img.shields.io/appveyor/ci/LaurentRDC/npstreams/master.svg
    :target: https://ci.appveyor.com/project/LaurentRDC/scikit-ued
    :alt: Windows Build Status
.. image:: https://readthedocs.org/projects/npstreams/badge/?version=master
    :target: http://scikit-ued.readthedocs.io
    :alt: Documentation Build Status
.. image:: https://img.shields.io/pypi/v/npstreams.svg
    :target: https://pypi.python.org/pypi/npstreams
    :alt: PyPI Version

Drop-in replacement of NumPy functions for streaming array operations.

Motivating Example
------------------

Consider the following snippet to combine 50 images 
from an iterable :data:`source`::

	import numpy as np

	images = np.empty( shape = (2048, 2048, 50) )
	from index, im in enumerate(source):
	    images[:,:,index] = im
	
	avg = np.average(images, axis = 2)

If the :data:`source` iterable provided 1000 images, the above routine would
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

While the :func:`average` example is simple, there are some functions that are not easily
brought 'online'. For example, the standard deviation is usually implemented as a two-pass algorithm,
but single-pass algorithms do exist and are implemented in this package.

API Reference
-------------

The `API Reference on readthedocs.io <http://scikit-ued.readthedocs.io>`_ provides API-level documentation, as 
well as tutorials.

Installation
------------

scikit-ued is available on PyPI; it can be installed with `pip <https://pip.pypa.io>`_.::

    python -m pip install npstreams

To install the latest development version from `Github <https://github.com/LaurentRDC/npstreams>`_::

    python -m pip install git+git://github.com/LaurentRDC/npstreams.git

Each version is tested against Python 3.5 and 3.6. If you are using a different version, tests can be run
using the standard library's `unittest` module.

Support / Report Issues
-----------------------

All support requests and issue reports should be
`filed on Github as an issue <https://github.com/LaurentRDC/npstreams/issues>`_.

License
-------

scikit-ued is made available under the BSD License, same as NumPy. For more details, see `LICENSE.txt <https://github.com/LaurentRDC/scikit-ued/blob/master/LICENSE.txt>`_.
