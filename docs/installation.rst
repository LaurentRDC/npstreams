.. include:: references.txt

.. _installation:

************
Installation
************

Requirements
============

**npstreams** works on Linux, Mac OS X and Windows. It requires Python 3.7+ 
as well as `numpy`_. `scipy`_ is an optional dependency that is only used in
tests; however, if SciPy cannot be imported, tests will not fail.

To get access to the :mod:`npstreams.cuda` module, which contains CUDA-enabled routines,
PyCUDA_ must be installed as well.

Install npstreams
=================

npstreams is available on PyPI; it can be installed with `pip <https://pip.pypa.io>`_::

    python -m pip install npstreams

npstreams can also be installed with the conda package manager, from the conda-forge channel::

    conda config --add channels conda-forge
    conda install npstreams

You can install the latest developer version of npstreams by cloning the git
repository::

    git clone https://github.com/LaurentRDC/npstreams.git

...then installing the package with::

    cd npstreams
    pip install .


Testing
=======

If you want to check that all the tests are running correctly with your Python
configuration, type::

    pip install .[development]
    pytest


Embedding in applications
=========================

`npstreams` is designed to be used in conjuction with multiprocessing libraries, such as the standard 
`multiprocessing` library. `npstreams` even uses `multiprocessing` directly in certain functions. 

In order to use the multicore functionality of `npstreams` in applications frozen with `py2exe`, `PyInstaller`, or `cx_Freeze`, 
you will need to activate the ``multiprocessing.freeze_support()`` function. `You can read more 
about it here. <https://docs.python.org/library/multiprocessing.html#multiprocessing.freeze_support>`_