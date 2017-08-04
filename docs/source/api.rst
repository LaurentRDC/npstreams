.. _api:

*************
Reference/API
*************

.. currentmodule:: npstreams

Click on any function below to see detailed information.

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
    isub

Linear Algebra
--------------
.. autosummary::
    :toctree: function/

    idot
    itensordot

Others
------
.. autosummary::
    :toctree: functions/

    iany
    iall
    istack
    iflatten

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

General Stream reduction
------------------------

You can assemble your own streaming reduction using the following generator:

    .. autofunction:: stream_reduce

This decorator will ensure that streams will be transformed into streams of NumPy arrays

    .. autofunction:: array_stream