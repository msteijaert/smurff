from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

SOURCES = ["smurff.pyx"]
INCLUDE_DIRS = []
LIBRARIES = ["blas", "smurff-cpp"]

setup(
    ext_modules = cythonize([
        Extension("smurff",
                  sources=SOURCES,
                  include_dirs=INCLUDE_DIRS,
                  libraries=LIBRARIES,
                  extra_compile_args=['-std=c++11'],
                  language="c++")
    ])
)
