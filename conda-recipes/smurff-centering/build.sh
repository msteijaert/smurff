#!/bin/bash

cd lib

wget http://bitbucket.org/eigen/eigen/get/3.3.4.tar.bz2
tar xjf 3.3.4.tar.bz2
rm -rf eigen3
mv eigen-eigen-5a0156e40feb eigen3

cd smurff-cpp/cmake
rm -rf build 
mkdir build
cd build
cmake ../ -DENABLE_OPENBLAS=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=${CONDA_PREFIX}  -DCMAKE_OSX_DEPLOYMENT_TARGET="" 
make 
make install
