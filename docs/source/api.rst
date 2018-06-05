.. include:: references.txt

.. _api:

*************
Reference/API
*************

.. currentmodule:: npstreams

Click on any function below to see detailed information.

Creation of Streams
-------------------

Decorator for streaming functions which guarantees that the stream elements will be converted to arrays.

.. autosummary::
    :toctree: functions/

    array_stream

The :func:`array_stream` decorator wraps iterables into an :class:`ArrayStream` iterator. This is not 
required to use the functions defined here, but it provides some nice guarantees.

.. autosummary::
    :toctree: classes/

    ArrayStream

Statistical Functions
---------------------
    
.. autosummary::
    :toctree: functions/

    imean
    iaverage
    istd
    ivar
    isem
    ihistogram

The following functions consume entire streams. By avoiding costly intermediate steps,
they can perform much faster than their generator versions.

.. autosummary::
    :toctree: functions/

    mean
    average
    std
    var
    sem
    
Numerics
--------

.. autosummary::
    :toctree: functions/

    isum
    iprod
    isub

.. autosummary::
    :toctree: functions/

    sum
    prod

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