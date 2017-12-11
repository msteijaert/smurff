#!/bin/bash

cd lib/smurff-cpp/cmake
rm -rf build
mkdir build
cd build
cmake ../ -DENABLE_OPENBLAS=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=${CONDA_PREFIX}  -DCMAKE_OSX_DEPLOYMENT_TARGET=""
make
make install
