#!/bin/bash


cd lib/smurff-cpp/cmake
rm -rf build 
mkdir build
cd build
cmake ../ -DENABLE_LAPACK=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=${CONDA_PREFIX} \
    -DENABLE_BOOST_RANDOM=ON \
    -DCMAKE_OSX_DEPLOYMENT_TARGET="" 
make -j${CPU_COUNT} 
make install
