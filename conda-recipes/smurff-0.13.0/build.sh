#!/bin/bash

rm -rf build 
mkdir build
cd build

cmake ../lib/smurff-cpp/cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$PREFIX \
    -DENABLE_MKL=ON -DOpenMP_CXX_FLAG=-fopenmp=libiomp5 -DOpenMP_C_FLAG=-fopenmp=libiomp5

make -j$CPU_COUNT
make install
cd python/Smurff

$PYTHON setup.py install --single-version-externally-managed --record=record.txt
