
Release 1.6.4
-------------

* Fixed an issue regarding a deprecation of `collections.Sized` (in favour of `collections.abc.Sized`) in Python 3.10+
* Code snippets in documentation are now tested for correctness.

Release 1.6.3
-------------

* Added support for Python 3.9

Release 1.6.2
-------------

* Added the ability to run default benchmarks from the command line with ``python -m npsteams.benchmarks``.
* Added explicit support for Python 3.8.
* Bumped requirement for `numpy >= 1.14`.

Release 1.6.1
-------------

* Added a changelog.
* Added the possibility to use weights in ``ihistogram``.
* Added the function ``average_and_var`` to compute the average and variance in a single pass.
* Documentation regarding the ``ddof`` keyword in many statistical wrongly stated that the default value was 1. This has been corrected. 

Release 1.6
-----------

* Fixed some issues with NumPy versions above 1.16.

Release 1.5.2
-------------

* Added benchmarking capabilities.
* Added the ``array_stream`` decorator.
* Removed support for Python < 3.6.