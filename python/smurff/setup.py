import subprocess

from pkg_resources import parse_version
from setuptools import setup
import Cython
from Cython.Distutils import Extension
from Cython.Build import cythonize
import numpy
import numpy.distutils.system_info as sysinfo
import sys
import os

if parse_version(Cython.__version__) < parse_version("0.25"):
    raise ValueError("Requires Cython >=0.25, have found Cython %s" % Cython.__version__)

lapack_opt_info = sysinfo.get_info("lapack_opt")
# {'libraries': ['mkl_rt', 'pthread'],
#  'library_dirs': ['/Users/vanderaa/miniconda3/lib'],
#  'define_macros': [('SCIPY_MKL_H', None), ('HAVE_CBLAS', None)],
#  'include_dirs': ['/Users/vanderaa/miniconda3/include']
# }

SOURCES            = ["smurff.pyx"]
INCLUDE_DIRS       = [ numpy.get_include() ]
LIBRARY_DIRS       = []
LIBRARIES          = ["smurff-cpp" ]
EXTRA_COMPILE_ARGS = ['-std=c++11']
EXTRA_LINK_ARGS    = []

if 'include_dirs' in lapack_opt_info:
    INCLUDE_DIRS += lapack_opt_info['include_dirs']

if 'library_dirs' in lapack_opt_info:
    LIBRARY_DIRS += lapack_opt_info['library_dirs']

if 'libraries' in lapack_opt_info:
    LIBRARIES += lapack_opt_info['libraries']

# add --with-smurff-cpp option
for arg in sys.argv:
    if arg.startswith("--with-smurff-cpp="):
        smurff_cpp_dir=arg[len("--with-smurff-cpp="):]
        INCLUDE_DIRS.append(os.path.join(smurff_cpp_dir, "include"))
        LIBRARY_DIRS.append(os.path.join(smurff_cpp_dir, "lib"))
        sys.argv.remove(arg)

# add cleanall option
for arg in sys.argv:
    if (arg == "cleanall"):
        print("Deleting cython files...")
        subprocess.Popen("rm -rf build *.cpp *.so", shell=True, executable="/bin/bash")
        sys.argv.remove(arg)
        sys.argv.append("clean")

# add --with-openmp option
for arg in sys.argv:
    if arg.startswith("--with-openmp"):
        openmp_lib=arg[len("--with-openmp"):]
        EXTRA_COMPILE_ARGS.append("-fopenmp" + openmp_lib)
        EXTRA_LINK_ARGS.append("-fopenmp" + openmp_lib)
        sys.argv.remove(arg)

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
    ext_modules = cythonize([
        Extension(
            "smurff",
                sources = SOURCES,
                include_dirs = INCLUDE_DIRS,
                libraries = LIBRARIES,
                library_dirs = LIBRARY_DIRS,
                extra_compile_args = EXTRA_COMPILE_ARGS,
                extra_link_args = EXTRA_LINK_ARGS,
                language = "c++",
            )
        ], compiler_directives={'c_string_type': 'str', 'c_string_encoding': 'default'}),
    # cythonize(ext_modules, compiler_directives={'c_string_type': 'str', 'c_string_encoding': 'default'}),
    classifiers = CLASSIFIERS,
    keywords = "bayesian factorization machine-learning high-dimensional side-information",
    install_requires = ['numpy', 'scipy', 'pandas']
)
