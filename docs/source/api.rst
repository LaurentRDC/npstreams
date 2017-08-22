.. include:: references.txt

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
    inansum
    iprod
    inanprod
    isub

CUDA-enabled routines
---------------------
`PyCUDA`_ must be installed. These routines must explicitly be imported from
the :mod:`npstreams.cuda` submodule.

.. autosummary::
    :toctree: functions/

    cuda.csum
    cuda.cprod
    cuda.caverage


Linear Algebra
--------------
.. autosummary::
    :toctree: functions/

    idot
    iinner
    itensordot
    ieinsum

Control Flow
------------
.. autosummary::
    :toctree: functions/

    ipipe

Comparisons
-----------
.. autosummary::
    :toctree: functions/

    iany
    iall
    imax
    imin

Stacking
--------
.. autosummary::
    :toctree: functions/

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
    peek

Parallelization
---------------
.. autosummary::
    :toctree: functions/

    pmap
    preduce