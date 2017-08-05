.. include:: references.txt

.. _making_your_own:

********************************************
Making your own Streaming Reduction Function
********************************************

.. currentmodule:: npstreams

============================================
The :func:`ireduce_ufunc` generator function
============================================

You can assemble your own streaming reduction function from a **binary** NumPy ufunc
using the following generator function:

    .. autofunction:: ireduce_ufunc

Note that while all NumPy ufuncs have a :meth:`reduce` method, not all of them are useful.
This is why :func:`ireduce_ufunc` will only work with **binary** ufuncs, most of which are listed below.

.. _numpy_binary_ufuncs:

===================
NumPy Binary Ufuncs
===================

:func:`ireduce_ufunc` is tested to work on the following binary ufuncs, which are available in `NumPy`_.


Arithmetics
-----------

.. autosummary::
    :nosignatures:

    numpy.add
    numpy.subtract
    numpy.multiply
    numpy.divide
    numpy.logaddexp
    numpy.logaddexp2
    numpy.true_divide
    numpy.floor_divide
    numpy.power
    numpy.remainder
    numpy.mod
    numpy.fmod

Trigonometric functions
-----------------------

.. autosummary::
    :nosignatures:

    numpy.arctan2
    numpy.hypot

Bit-twiddling functions
-----------------------

.. autosummary::
    :nosignatures:

    numpy.bitwise_and
    numpy.bitwise_or
    numpy.bitwise_xor
    numpy.left_shift
    numpy.right_shift

Comparison functions
--------------------

.. autosummary::
    :nosignatures:

    numpy.greater
    numpy.greater_equal
    numpy.less
    numpy.less_equal
    numpy.not_equal
    numpy.equal
    numpy.logical_and
    numpy.logical_or
    numpy.logical_xor
    numpy.maximum
    numpy.fmax
    numpy.minimum
    numpy.fmin

Floating functions
------------------

.. autosummary::
    :nosignatures:

    numpy.copysign
    numpy.nextafter
    numpy.ldexp

================
Stream of arrays
================

This decorator will ensure that streams will be transformed into streams of NumPy arrays.
A single NumPy array can be passed to a function expecting a stream, decorated with :func:`array_stream`;
this solitary array will be repackaged into a sequence of length one.

    .. autofunction:: array_stream

==========================
Example: Streaming Maximum
==========================

Let's create a streaming maximum function for a stream. First, we have to choose 
how to handle NaNs:

* If we want to propagate NaNs, we should use :func:`numpy.maximum`
* If we want to ignore NaNs, we should use :func:`numpy.fmax`

Both of those functions are binary ufuncs, so we can use :func:`ireduce_ufunc`. We will
also want to make sure that anything in the stream that isn't an array will be made into one
using the :func:`array_stream` decorator.

Putting it all together::

    from npstreams import array_stream, ireduce_ufunc
    from numpy import maximum, fmax

    @array_stream
    def imax(arrays, axis = -1, ignore_nan = False, **kwargs):
        """
        Streaming maximum along an axis.

        Parameters
        ----------
        arrays : iterable
            Stream of arrays to be compared.
        axis : int or None, optional
            Axis along which to compute the maximum. If None, 
            arrays are flattened before reduction.
        ignore_nan : bool, optional
            If True, NaNs are ignored. Default is False.
        
        Yields
        ------
        online_max : ndarray
        """
        ufunc = fmax if ignore_nan else maximum
        yield from ireduce_ufunc(arrays, ufunc, axis = axis, **kwargs)

This will provide us with a streaming function, meaning that we can look at the progress
as it is being computer. We can also create a function that returns the max of the stream
like :meth:`numpy.ndarray.max()` using the :func:`npstreams.last` function::

    from npstreams import last

    def smax(*args, **kwargs):  # s for stream
        """
        Maximum of all arrays in a stream, along an axis.

        Parameters
        ----------
        arrays : iterable
            Stream of arrays to be compared.
        axis : int or None, optional
            Axis along which to compute the maximum. If None, 
            arrays are flattened before reduction.
        ignore_nan : bool, optional
            If True, NaNs are ignored. Default is False.
        
        Returns
        -------
        max : scalar or ndarray
        """
        return last(imax(*args, **kwargs)