.. include:: references.txt

.. _npstreams:

**************************************
`npstreams`: streaming NumPy functions
**************************************

npstreams is an open-source Python package for streaming NumPy array operations. 
The goal is to provide tested, drop-in replacements for NumPy functions (where possible) 
that operate on streams of arrays instead of dense arrays.

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

Recipe: averaging with error
------------------------------

It is possible to combine :func:`imean` and :func:`isem` into a single stream using :func:`itertools.tee`. 
Here is a recipe for it::

    from itertools import tee
    from npstreams import imean, isem

	def iaverage_with_error(arrays):    
	    """ 
	    Combined streaming mean and standard error of arrays. 
		
	    Parameters
	    ----------
	    arrays : iterable of ndarrays
	    	Arrays to be averaged. This iterable can also a generator.
	    weights : iterable of ndarray, iterable of floats, or None, optional
	    	Array of weights. See `numpy.average` for further information.
	    
	    Yields
	    ------
	    avg : `~numpy.ndarray`
	    	Weighted average. 
	    sem : `~numpy.ndarray`
	    	Standard error in the mean
	    """
	    stream1, stream2 = itertools.tee(arrays, 2)
	    yield from zip(imean(stream1), isem(stream2))

Links
=====

* `Source code <https://github.com/LaurentRDC/npstreams>`_
* `Issues <https://github.com/LaurentRDC/npstreams/issues>`_
* `Docs <http://npstreams.readthedocs.org/>`_

.. _npstreams_docs:

General Documentation
=====================

.. toctree::
    :maxdepth: 2
    
    installation
    api

Authors
=======

* Laurent P. René de Cotret