import sys
import numpy as np
from glob import glob
import subprocess


from distutils.command.build_clib import build_clib
from distutils.errors    import DistutilsSetupError
from distutils.sysconfig import get_python_inc
from setuptools          import setup
from setuptools          import Extension
from datetime            import datetime

import Cython
from Cython.Distutils import build_ext
from Cython.Build import cythonize

import os
from textwrap import dedent

# for downloading Eigen
import tempfile
import urllib
import tarfile
import shutil

# checking out libfastsparse
import subprocess

blas_libs = ['blas', 'lapack']
macau_libs = ['smurff-cpp']
inc = ['lib/smurff-cpp', 'lib/eigen3', 'lib/libfastsparse', np.get_include(), get_python_inc()]
ldirs = ['lib/smurff-cpp']

ext_modules=[
    Extension("smurff.smurff",
              sources = ["python/smurff/smurff.pyx"],
              include_dirs = inc,
              libraries = blas_libs + macau_libs,
              library_dirs = ldirs,
              runtime_library_dirs = ldirs,
              extra_compile_args = ['-std=c++11', '-g'],
              language = "c++")
]

CLASSIFIERS = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT",
    "Programming Language :: C++",
    "Programming Language :: Python",
    "Programming Language :: Python :: 2.7",
    "Programming Language :: Python :: 3",
    "Topic :: Machine Learning",
    "Topic :: Matrix Factorization",
    "Operating System :: Microsoft :: Windows",
    "Operating System :: POSIX",
    "Operating System :: Unix",
    "Operating System :: MacOS"
]

def main():
    setup(
        name = 'smurff',
        version = subprocess.check_output("git describe", shell=True).rstrip(),
        packages = ["smurff"],
        package_dir = {'' : 'python'},
        url = "http://github.com/jaak-s/smurff",
        license = "MIT",
        description = 'Bayesian Factorization Methods',
        long_description = 'Highly optimized and parallelized methods for Bayesian Factorization, including BPMF and smurff. The package uses optimized OpenMP/C++ code with a Cython wrapper to factorize large scale matrices. smurff method provides also the ability to incorporate high-dimensional side information to the factorization.',
        author = "Jaak Simm",
        author_email = "jaak.simm@gmail.com",
        cmdclass = {'build_clib': build_clib, 'build_ext': build_ext},
        ext_modules = cythonize(ext_modules, include_path=sys.path),
        classifiers = CLASSIFIERS,
        keywords = "bayesian factorization machine-learning high-dimensional side-information",
        install_requires=['numpy', 'scipy', 'pandas']
    )

if __name__ == '__main__':
    main()

