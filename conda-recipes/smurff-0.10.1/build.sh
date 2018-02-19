#!/bin/bash

cd lib/smurff-cpp/cmake
rm -rf build 
mkdir build
cd build
cmake ../ -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=${CONDA_PREFIX}  -DCMAKE_OSX_DEPLOYMENT_TARGET="" \
     -DBOOST_ROOT=${CONDA_PREFIX} -DMPI_C_FOUND=NO

make 
make install
