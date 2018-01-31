#!/bin/bash

cd lib/smurff-cpp/cmake
rm -rf build && mkdir build && cd build
cmake ../ -DENABLE_OPENBLAS=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="${PREFIX}" 
make -j${CPU_COUNT}
make install
