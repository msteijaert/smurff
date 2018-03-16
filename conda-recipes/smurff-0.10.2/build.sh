#!/bin/bash

test $(uname) == "Darwin" && CMAKE_OPENMP_FLAGS="-DOpenMP_CXX_FLAGS=-fopenmp=libiomp5 -DOpenMP_C_FLAGS=-fopenmp=libiomp5"
 
pushd lib/smurff-cpp/cmake
rm -rf build 
mkdir build
cd build
cmake ../ -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$PREFIX \
    -DENABLE_OPENBLAS=ON -DENABLE_LAPACK=OFF ${CMAKE_OPENMP_FLAGS}
make -j$CPU_COUNT
make install
popd

pushd python/smurff
$PYTHON setup.py install --with-openmp --single-version-externally-managed --record=record.txt
popd
