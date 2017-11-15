import subprocess

from setuptools import setup
from setuptools import Extension
from Cython.Build import cythonize

SOURCES = ["smurff.pyx"]
INCLUDE_DIRS = []
LIBRARIES = ["blas", "smurff-cpp"]

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

ext_modules = [
    Extension(
        "smurff",
        sources = SOURCES,
        include_dirs = INCLUDE_DIRS,
        libraries = LIBRARIES,
        extra_compile_args = ['-std=c++11'],
        language = "c++")
]

setup(
    name = 'smurff',
    version = subprocess.check_output("git describe", shell=True).rstrip().decode(),
    # packages = ["smurff"],
    # package_dir = {'' : 'python'},
    url = "http://github.com/ExaScience/smurff",
    license = "MIT",
    description = 'Bayesian Factorization Methods',
    long_description = 'Highly optimized and parallelized methods for Bayesian Factorization, including BPMF and smurff. The package uses optimized OpenMP/C++ code with a Cython wrapper to factorize large scale matrices. smurff method provides also the ability to incorporate high-dimensional side information to the factorization.',
    author = "Tom Vander Aa",
    author_email = "Tom.VanderAa@imec.be",
    ext_modules = cythonize(ext_modules),
    classifiers = CLASSIFIERS,
    keywords = "bayesian factorization machine-learning high-dimensional side-information",
    install_requires = ['numpy', 'scipy', 'pandas']
)
