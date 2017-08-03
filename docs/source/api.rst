.. _api:

*************
Reference/API
*************

.. currentmodule:: npstreams

Naming Conventions
==================

In order to facilitate documentation, functions in :mod:`npstreams` follow the following conventions:

    * Routines are named after their closest equivalent in :mod:`numpy` and :mod:`scipy`.
    * Routines with names starting with 'i' (e.g. :func:`iprod`) return a generator.
    * Routines with names starting with 'p' (e.g. :func:`pmap`) can be parallelized. The default
      behavior is always to not use multiple cores.

Axis Conventions
================

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

Examples
--------

TODO


Reference
=========
Click on any function below to see detailed information.

Iterator Utilities
------------------
.. autosummary::
    :toctree: functions/

    last
    chunked
    linspace
    multilinspace

Parallelization
---------------
.. autosummary::
    :toctree: functions/

    pmap
    preduce

Statistical Functions
---------------------
.. autosummary::
    :toctree: functions/

    imean
    inanmean
    iaverage
    istd
    inanstd
    ivar
    inanvar
    isem


Numerics
--------
.. autosummary::
    :toctree: functions/

    isum
    psum
    inansum
    iprod
    pprod
    inanprod

General Stream reduction
------------------------

You can assemble your own streaming reduction using the following generator:

    .. autofunction:: stream_reduce