.. include:: references.txt

.. _installation:

************
Installation
************

Requirements
============

**npstreams** works on Linux, Mac OS X and Windows. It requires Python 3.4+ 
as well as `numpy`_. `scipy`_ is an optional dependency that is only used in
tests; however, if SciPy cannot be imported, tests will not fail.

To get access to the :mod:`npstreams.cuda` module, which contains CUDA-enabled routines,
PyCUDA_ must be installed as well.

Install npstreams
=================

You can install the latest developer version of npstreams by cloning the git
repository::

    git clone https://github.com/LaurentRDC/npstreams.git

...then installing the package with::

    cd npstreams
    python setup.py install


Testing
=======

If you want to check that all the tests are running correctly with your Python
configuration, type::

    python setup.py test