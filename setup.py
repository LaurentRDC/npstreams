# -*- coding: utf-8 -*-
import os
import re
from glob import glob

from setuptools import Extension, find_packages, setup

# from Cython.Build import cythonize
# import numpy

# To upload to pypi.org:
#   >>> python setup.py sdist
#   >>> twine upload dist/npstreams-x.x.x.tar.gz

BASE_PACKAGE = "npstreams"

base_path = os.path.dirname(__file__)
with open(os.path.join(base_path, "npstreams", "__init__.py")) as f:
    module_content = f.read()
    VERSION = (
        re.compile(r".*__version__ = \"(.*?)\"", re.S).match(module_content).group(1)
    )
    LICENSE = (
        re.compile(r".*__license__ = \"(.*?)\"", re.S).match(module_content).group(1)
    )


with open("README.md") as f:
    readme = f.read()

with open("requirements.txt") as f:
    requirements = [line for line in f.read().split("\n") if len(line.strip())]

exclude = {"exclude": ["external*", "docs", "*cache"]}
packages = [
    BASE_PACKAGE + "." + x
    for x in find_packages(os.path.join(base_path, BASE_PACKAGE), **exclude)
]
if BASE_PACKAGE not in packages:
    packages.append(BASE_PACKAGE)


def test_suite():
    return TestLoader().discover(".")


if __name__ == "__main__":
    setup(
        name="npstreams",
        description="Streaming operations on NumPy arrays",
        long_description=readme,
        long_description_content_type="text/markdown",
        license=LICENSE,
        url="",
        download_url="http://github.com/LaurentRDC/npstreams",
        version=VERSION,
        author="Laurent P. René de Cotret",
        author_email="laurent.renedecotret@mail.mcgill.ca",
        maintainer="Laurent P. René de Cotret",
        maintainer_email="laurent.renedecotret@mail.mcgill.ca",
        install_requires=requirements,
        keywords=["streaming", "numpy", "math"],
        packages=packages,
        include_package_data=True,
        python_requires=">=3.7",
        zip_safe=False,
        #        include_dirs = [numpy.get_include()],
        #        ext_modules = cythonize("npstreams/*.pyx",
        #                                 compiler_directives = {'language_level':3,
        #                                                        'boundscheck': False}),
        classifiers=[
            "Environment :: Console",
            "Intended Audience :: Science/Research",
            "Topic :: Scientific/Engineering",
            "License :: OSI Approved :: BSD License",
            "Natural Language :: English",
            "Operating System :: OS Independent",
            "Programming Language :: Python",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
        ],
    )
