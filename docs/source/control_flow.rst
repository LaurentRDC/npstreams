.. include:: references.txt

.. _control_flow:

************
Control Flow
************

.. currentmodule:: npstreams

=========================
Streaming array pipelines
=========================

Before reducing your stream of arrays (e.g. averaging them together), you may want to 
transform them. This can be done with the :func:`ipipe` function:

.. autofunction:: ipipe
    :noindex:

Imagine we have the following pipeline, in which we want processes images in some iterable :data:`arrays` 
as follows:

* Remove negative pixel intensity values;
* Adjust the gamma value of images (from Scikit-image's :mod:`exposure` module);
* Average the result together.

The following lines will do the trick::

    from functools import partial
    from npstreams import ipipe, iaverage, last
    from skimage.exposure import adjust_gamma

    def remove_negative(arr):
        arr[arr < 0] = 0
        return arr

    pipeline = ipipe(adjust_gamma, remove_negative, arrays)
    avgs = last(iaverage(pipeline))

If the pipeline is computationally intensive, we can also pipe arrays in parallel using the 
keyword-only ``processes``::

    pipeline = ipipe(adjust_gamma, remove_negative, arrays, processes = 4)  # 4 cores will be used
    avgs = last(iaverage(pipeline))

Since :func:`ipipe` uses :func:`pmap` under the hood, we can also use all available cores
by passing ``processes = None``