#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import io
import setuptools


def read(fname):
    file_path = os.path.join(os.path.dirname(__file__), fname)
    return io.open(file_path, encoding="utf-8").read()


package_name = "pointpp"

version = None
init_py = os.path.join(package_name.replace("-", "_"), "__init__.py")
for line in read(init_py).split("\n"):
    if line.startswith("__version__"):
        version = line.split("=")[-1].strip()[1:-1]
assert version

setuptools.setup(
    name=package_name,
    version=version,
    description='Program to post-process verif files',
    url='https://github.com/tnipen/pointpp',
    author='Thomas Nipen',
    author_email='thomas.nipen@met.no',
    packages=setuptools.find_packages(exclude=["test"]),
    license='BSD-3',
    install_requires=['numpy>=1.7', 'matplotlib', 'scipy', 'netCDF4', 'verif>=1.0.0', 'sklearn', 'gridpp>=0.6.0'],
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 4 - Beta',

        # Indicate who your project is intended for
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Atmospheric Science',
        'Topic :: Scientific/Engineering :: Information Analysis',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: BSD License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],

    # What does your project relate to?
    keywords='meteorology post-processing weather prediction',

    extras_require={
        "test": ["coverage", "pep8"],
    },

    test_suite="pointpp.tests",

    # To provide executable scripts, use entry points in preference to the
    # "scripts" keyword. Entry points provide cross-platform support and allow
    # pip to create the appropriate form of executable for the target platform.
    entry_points={
        'console_scripts': [
            'pointpp=pointpp:main',
            'pointgen=pointpp:pointgen',
        ],
    },
)
