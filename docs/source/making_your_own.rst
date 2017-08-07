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

The non-generator version is also available:

    .. autofunction:: reduce_ufunc

Note that while all NumPy ufuncs have a :meth:`reduce` method, not all of them are useful.
This is why :func:`ireduce_ufunc` and :func:`reduce_ufunc` will only work with **binary** ufuncs, most of which are listed below.

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

==========================
Example: Streaming Maximum
==========================

Let's create a streaming maximum function for a stream. First, we have to choose 
how to handle NaNs:

* If we want to propagate NaNs, we should use :func:`numpy.maximum`
* If we want to ignore NaNs, we should use :func:`numpy.fmax`

Both of those functions are binary ufuncs, so we can use :func:`ireduce_ufunc`. Note that any function based
on :func:`ireduce_ufunc` or :func:`reduce_ufunc` will automatically work on streams of numbers thanks to the
 :func:`array_stream` decorator.

Putting it all together::

    from npstreams import ireduce_ufunc
    from numpy import maximum, fmax

    def imax(arrays, axis = -1, ignore_nan = False, **kwargs):
        """
        Streaming cumulative maximum along an axis.

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
as it is being computed. We can also create a function that returns the max of the stream
like :meth:`numpy.ndarray.max()` using the :func:`reduce_ufunc` function::

    from npstreams import reduce_ufunc

    def smax(*args, **kwargs):  # s for stream
        """
        Maximum of a stream along an axis.

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
        max : ndarray
        """
        return reduce_ufunc(*args, **kwargs)