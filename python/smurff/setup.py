from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

SOURCES = ["smurff.pyx"]
INCLUDE_DIRS = ["../../lib/smurff-cpp", "../../lib/eigen3"]
LIBRARIES = ["blas"]

setup(
   ext_modules = cythonize([
		Extension("smurff",
				    sources=SOURCES,
                include_dirs = INCLUDE_DIRS,
                libraries=LIBRARIES,
                extra_objects=["../../lib/smurff-cpp/cmake/build/libraries/SmurffCpp/libsmurff-cpp.a"],
                extra_compile_args=['-std=c++11'],
                language="c++")
      ])
)
