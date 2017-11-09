from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

SOURCES = ["/home/ipasechnikov/smurff-python/smurff.pyx"]
INCLUDE_DIRS = ["/home/ipasechnikov/smurff/lib/smurff-cpp", "/home/ipasechnikov/smurff/lib/eigen3"]
LIBRARIES = ["blas"]

setup(
   ext_modules = cythonize([
		Extension("smurff",
				    sources=SOURCES,
                include_dirs = INCLUDE_DIRS,
                libraries=LIBRARIES,
                extra_objects=["/home/ipasechnikov/smurff/lib/smurff-cpp/cmake/build/libraries/SmurffCpp/libsmurff-cpp.a"],
                extra_compile_args=['-std=c++11'],
                language="c++")
      ])
)
