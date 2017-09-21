#!/bin/bash

cd lib/macau-cpp/openmp

export CPATH=$CPATH:$PREFIX/include/eigen3
export CPATH=$CPATH:/usr/local/include
export LIBRARY_PATH=$LIBRARY_PATH:$PREFIX/lib
export LIBRARY_PATH=$LIBRARY_PATH:/usr/local/lib

sed -i -e 's/^LIBLOCS/#LIBLOCS/g' Makefile
sed -i -e 's/^CXX/#CXX/g' Makefile
#sed -i -e '/argp/d' Makefile

if [ -n "$OSX_ARCH" ]
then
    make -j CXX=g++-6
else 
    make -j
fi

install cmd_nompi $PREFIX/bin/smurff
