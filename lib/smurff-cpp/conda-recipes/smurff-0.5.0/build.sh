#!/bin/bash

cd lib/macau-cpp/openmp
export CPATH=$CPATH:$PREFIX/include/eigen3
export LIBRARY_PATH=$LIBRARY_PATH:$PREFIX/lib
sed -i -e 's/^LIBLOCS/#LIBLOCS/g' Makefile
#sed -i -e 's/^CXX/#CXX/g' Makefile
#sed -i -e '/argp/d' Makefile

make -j
install cmd_nompi $PREFIX/bin
