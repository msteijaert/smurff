#!/bin/bash

set -x

rm -rf build && mkdir build && cd build

cmake ../lib/smurff-cpp/cmake -DCMAKE_BUILD_TYPE=Release \
    -DBoost_USE_STATIC_LIBS=OFF \
    -DCMAKE_INSTALL_PREFIX=$PREFIX \
    -DCMAKE_PREFIX_PATH=$PREFIX \
    -DENABLE_MKL=ON \
    -DCMAKE_C_COMPILER=$GCC \
    -DCMAKE_CXX_COMPILER=$GXX

make -j$CPU_COUNT
make install
cd python/Smurff
$PYTHON setup.py install --single-version-externally-managed --record=record.txt
