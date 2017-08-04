.. include:: references.txt

.. _installation:

************
Installation
************

Requirements
============

.. note::

    Users are strongly recommended to manage these dependencies with the
    excellent `Intel Distribution for Python <https://software.intel.com/en-us/intel-distribution-for-python>`_
    which provides easy access to all of the above dependencies and more.

**npstreams** works on Linux, Mac OS X and Windows. It requires Python 3.4+ 
as well as `numpy`_. `scipy`_ is an optional dependency that is only used in
tests.

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
