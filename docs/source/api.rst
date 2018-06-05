.. include:: references.txt

.. _api:

*************
Reference/API
*************

.. currentmodule:: npstreams

Click on any function below to see detailed information.

Creation of Streams
-------------------

.. autosummary::
    :toctree: functions/

    array_stream

The :func:`array_stream` decorator wraps iterables into an :class:`ArrayStream` iterator:

.. autosummary::
    :toctree: classes/

    ArrayStream

Statistical Functions
---------------------

.. autosummary::
    :toctree: functions/

    mean
    average
    std
    var
    sem
    
.. autosummary::
    :toctree: functions/

    imean
    iaverage
    istd
    ivar
    isem
    ihistogram
    
Numerics
--------
.. autosummary::
    :toctree: functions/

    sum
    prod

.. autosummary::
    :toctree: functions/

    isum
    iprod
    isub

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
    iload
    pload

Comparisons
-----------
.. autosummary::
    :toctree: functions/

    iany
    iall
    imax
    imin

Parallelization
---------------
.. autosummary::
    :toctree: functions/

    pmap
    pmap_unordered
    preduce

Stacking
--------
.. autosummary::
    :toctree: functions/

    stack
    
Iterator Utilities
------------------
.. autosummary::
    :toctree: functions/

    last
    cyclic
    itercopy
    chunked
    linspace
    multilinspace
    peek
    primed

Array Utilities
---------------
.. autosummary::
    :toctree: functions/

    nan_to_num