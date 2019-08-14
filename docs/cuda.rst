.. include:: references.txt

.. _cuda:

============
CUDA support
============

.. currentmodule:: npstreams

What is CUDA
============

`CUDA <http://nvidia.com/cuda/>`_ is a computing platform taking advantage of Nvidia hardware. 
It effectively allows for array computations on Graphical Processing Units (GPU).

:mod:`npstreams` relies on the (optional) `PyCUDA`_ library
to access CUDA functionality.

Advantages of CUDA
------------------

TODO: benchmarks

CUDA in npstreams
=================

`PyCUDA`_ is an optional dependency. Therefore, the CUDA-enabled functions are located in a separate
module, the :mod:`npstreams.cuda` submodule. 

Importing from :mod:`npstreams.cuda` submodule
----------------------------------------------

Importing anything from the :mod:`npstreams.cuda` submodule will raise an ``ImportError`` in the following cases:

    * `PyCUDA`_ is not installed;
    * No GPUs are available;
    * CUDA compilation backend is not available, possibly due to incomplete installation.

With this in mind, it is wise to wrap import statements from :mod:`npstreams.cuda` in a ``try/except`` block.

CUDA-enabled routines
---------------------

A limited set of functions implemented in npstreams also have CUDA-enabled equivalents. For performance reasons,
all CUDA-enabled routines operate along the 'stream' axis, i.e. as if the arrays had been stacked 
along a new dimension.