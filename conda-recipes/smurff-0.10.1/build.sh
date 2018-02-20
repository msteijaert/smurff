#!/bin/bash

pushd lib/smurff-cpp/cmake
rm -rf build 
mkdir build
cd build
cmake ../ -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=${CONDA_PREFIX}  -DCMAKE_OSX_DEPLOYMENT_TARGET=""
make 
make install
popd

pushd python/smurff
python setup.py install --with-openmp --single-version-externally-managed --record=record.txt
popd
