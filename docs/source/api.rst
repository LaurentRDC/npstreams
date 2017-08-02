.. _api:

*************
Reference/API
*************

.. currentmodule:: npstreams

Axis Conventions
================

NumPy arrays provide operations along axes. Similarly, :mod:`npstreams` also
exposes the :data:`axis` keyword in some (most?) reduction functions like :func:`isum`
and :func:`iprod`.

The convention for specification of the :data:`axis` parameter is as follows:

* The default (`axis = -1`) always corresponds to combining arrays along a
new axis. For example, summing images together along `axis = -1` is equivalent
to stacking images along a new axis, then averaging along this new axis

* If `axis = None`, arrays are flattened before being combined. The result will
be a single number.

* if `axis` is an `int`, then arrays are reduced according to this axis, and then combined.

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