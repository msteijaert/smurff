#!/bin/bash

set -e

# VERSION=v0.5.0-excape
VERSION=stable
PREFIX=
MODE=openmp


while [[ $# -gt 1 ]]
do
    key="$1"

    case $key in
        --version)
            VERSION="$2"
            shift # past argument
            ;;
        --prefix)
            PREFIX="$2"
            shift # past argument
            ;;
        --mode)
            MODE="$2"
            shift # past argument
            ;;
         *)
            # unknown option
            ;;
    esac
    shift # past argument or value
done


test -z "$PREFIX" && PREFIX=$(mktemp -d -t smurff-v${VERSION}-)
mkdir -p $PREFIX

echo "Installing smurff $VERSION in $PREFIX"

cd $PREFIX
PREFIX=$PWD

rm -rf smurff
git clone https://github.com/ExaScience/macau smurff
cd smurff
git submodule init
git submodule update
cd lib

wget -qO - http://bitbucket.org/eigen/eigen/get/3.3.3.tar.bz2 | tar xjf -
mv eigen-eigen-* eigen3

git checkout $VERSION

cd macau-cpp/$MODE

if [ -z "$CXX" ]; then
    make -j
else 
    make -j CXX="$CXX"
fi

set -x

mkdir -p $PREFIX/bin

cp cmd_nompi $PREFIX/bin/smurff


echo "export PATH=\$PATH:$PREFIX/bin"
