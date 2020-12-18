# npstreams

[![Documentation Build Status](https://readthedocs.org/projects/npstreams/badge/?version=master)](http://npstreams.readthedocs.io) [![PyPI Version](https://img.shields.io/pypi/v/npstreams.svg)](https://pypi.python.org/pypi/npstreams) [![Conda-forge Version](https://img.shields.io/conda/vn/conda-forge/npstreams.svg)](https://anaconda.org/conda-forge/npstreams)

npstreams is an open-source Python package for streaming NumPy array
operations. The goal is to provide tested routines that operate on
streams (or generators) of arrays instead of dense arrays.

Streaming reduction operations (sums, averages, etc.) can be implemented
in constant memory, which in turns allows for easy parallelization.

This approach has been a huge boon when working with lots of images; the
images are read one-by-one from disk and combined/processed in a
streaming fashion.

This package is developed in conjunction with other software projects in
the [Siwick research group](http://www.physics.mcgill.ca/siwicklab/).

## Motivating Example

Consider the following snippet to combine 50 images from an iterable
`source`:

```python
import numpy as np

images = np.empty( shape = (2048, 2048, 50) )
for index, im in enumerate(source):
    images[:,:,index] = im

avg = np.average(images, axis = 2)
```

If the `source` iterable provided 1000 images, the above routine would
not work on most machines. Moreover, what if we want to transform the
images one by one before averaging them? What about looking at the
average while it is being computed? Let\'s look at an example:

```python
import numpy as np
from npstreams import iaverage
from scipy.misc import imread

stream = map(imread, list_of_filenames)
averaged = iaverage(stream)
```

At this point, the generators `map` and `iaverage` are \'wired\' but
will not compute anything until it is requested. We can look at the
average evolve:

```python
import matplotlib.pyplot as plt
for avg in average:
    plt.imshow(avg); plt.show()
```

We can also use `last` to get at the final average:

```python
from npstreams import last

total = last(averaged) # average of the entire stream
```

## Streaming Functions

npstreams comes with some streaming functions built-in. Some examples:

-   Numerics : `isum`, `iprod`, `isub`, etc.
-   Statistics : `iaverage` (weighted mean), `ivar` (single-pass
    variance), etc.

More importantly, npstreams gives you all the tools required to build
your own streaming function. All routines are documented in the [API
Reference on readthedocs.io](http://npstreams.readthedocs.io).

## Benchmarking

npstreams provides a function for benchmarking common use cases.

To run the benchmark with default parameters, from the interpreter:

```python
from npstreams import benchmark
benchmark()
```

From a command-line terminal:

```bash
python -c 'import npstreams; npstreams.benchmark()'
```

The results will be printed to the screen.

## Future Work

Some of the features I want to implement in this package in the near
future:

-   Optimize the CUDA-enabled routines
-   More functions : more streaming functions borrowed from NumPy and
    SciPy.

## API Reference

The [API Reference on readthedocs.io](http://npstreams.readthedocs.io)
provides API-level documentation, as well as tutorials.

## Installation

The only requirement is NumPy. To have access to CUDA-enabled routines,
PyCUDA must also be installed. npstreams is available on PyPI; it can be
installed with [pip](https://pip.pypa.io).:

```bash
python -m pip install npstreams
```

npstreams can also be installed with the conda package manager, from the
conda-forge channel:

```bash
conda config --add channels conda-forge
conda install npstreams
```

To install the latest development version from
[Github](https://github.com/LaurentRDC/npstreams):

```bash
python -m pip install git+git://github.com/LaurentRDC/npstreams.git
```

Tests can be run using the `pytest` package.

## Citations

If you find this software useful, please consider citing the following
publication:

> L. P. René de Cotret, M. R. Otto, M. J. Stern. and B. J. Siwick, *An open-source software ecosystem for the interactive exploration of ultrafast electron scattering data*, Advanced Structural and Chemical Imaging 4:11 (2018) [DOI: 10.1186/s40679-018-0060-y.](https://ascimaging.springeropen.com/articles/10.1186/s40679-018-0060-y)


## Support / Report Issues

All support requests and issue reports should be [filed on Github as an
issue](https://github.com/LaurentRDC/npstreams/issues).

## License

npstreams is made available under the BSD License, same as NumPy. For
more details, see
[LICENSE.txt](https://github.com/LaurentRDC/npstreams/blob/master/LICENSE.txt).
