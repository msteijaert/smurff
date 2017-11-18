#!/bin/sh

set -e

[ $# != 1 ] && ( echo "Usage $0 <install_prefix>"; exit -1 )

PREFIX=$1
VERSION=3.3.4

mkdir -p $PREFIX
cd $PREFIX
PREFIX=$PWD


wget http://bitbucket.org/eigen/eigen/get/${VERSION}.tar.gz
tar xzf ${VERSION}.tar.gz
rm ${VERSION}.tar.gz
cd eigen-eigen-*
mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=$PREFIX
make install
