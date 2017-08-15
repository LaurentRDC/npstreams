.. include:: references.txt

.. _conventions:

***********
Conventions
***********

.. currentmodule:: npstreams

Stream Conventions
------------------

Most (all?) functions in :mod:`npstreams` are designed to work on streams, or
iterables of NumPy arrays. These iterables can be infinite. 
The quintessential example is a stream of images progressively read from disk. 
These streams of arrays must contain arrays that all have the same shape and data-type, 
unless specified otherwise. 

An example of a function that operates on a stream of arrays of different shapes is :func:`ieinsum`

A single NumPy array can be passed where a stream is expected; the array will be repackaged
into a stream of a single array.

Naming Conventions
------------------

In order to facilitate documentation, functions in :mod:`npstreams` follow the following conventions:

    * Routines are named after their closest equivalent in :mod:`numpy` and :mod:`scipy`.
    * Routines with names starting with 'i' (e.g. :func:`iprod`) return a generator.
    * Routines with names starting with 'p' (e.g. :func:`pmap`) can be parallelized. The default
      behavior is always to not use multiple cores.

Axis Conventions
----------------

NumPy arrays provide operations along axes. Similarly, :mod:`npstreams` also
exposes the :data:`axis` keyword in some (most?) reduction functions like :func:`isum`
and :func:`iprod`.

The convention for specification of the :data:`axis` parameter is as follows:

    * If ``axis = None``, arrays are flattened before being combined. The result will
      be a scalar of a 0d array.
    * The default (``axis = -1``) always corresponds to combining arrays along a
      new axis. For example, summing images together along ``axis = -1`` is equivalent
      to stacking images along a new axis, then averaging along this new axis
    * if ``axis`` is an ``int``, then arrays are reduced according to this axis, and then combined.
