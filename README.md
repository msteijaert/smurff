# SMURFF - Scalable Matrix Factorization Framework
[![Build Status](https://travis-ci.org/ExaScience/smurff.svg?branch=master)](https://travis-ci.org/ExaScience/smurff)

## Bayesian Factorization with Side Information

Highly optimized and parallelized methods for Bayesian Factorization, including BPMF, Macau and GFA.
The package uses optimized OpenMP/C++ code with a Cython wrapper to factorize large scale matrices.

Macau method is able to perform **matrix** and **tensor** factorization while incorporating high-dimensional side information to the factorization.

# Examples
For examples see [documentation](http://macau.readthedocs.io/en/latest/source/examples.html).

# Installation
Currently there is only C++ version of smurff available. Python one is comming soon.

## Source installation on Ubuntu
```bash
# install dependencies:
sudo apt-get install libopenblas-dev autoconf gfortran

# checkout and install Smurff:
git clone https://github.com/ExaScience/smurff.git
cd smurff
git checkout smurff
git submodule init
git submodule update
cd lib/smurff-cpp/makefiles/linux
make

# test Smurff:
wget http://homes.esat.kuleuven.be/~jsimm/chembl-IC50-346targets.mm
./smurff --train chembl-IC50-346targets.mm
```

## Source installation on Ubuntu using cmake
Before continuing installation please check that cmake version is at least 3.6

This is required due to the fixed Find scripts for BLAS libraries that are present in latest version.

cmake has multiple switches:

Build type switches:
* CMAKE_BUILD_TYPE - Debug/Release

Algebra library switches (select only one):
* ENABLE_BLAS - ON/OFF
* ENABLE_LAPACK - ON/OFF
* ENABLE_OPENBLAS - ON/OFF

```bash
# install dependencies:
sudo apt-get install libopenblas-dev autoconf gfortran

# checkout and install Smurff
git clone https://github.com/ExaScience/smurff.git
cd smurff
git checkout smurff
git submodule init
git submodule update
cd lib/smurff-cpp/
mkdir build
cd build
cmake ../ -DENABLE_OPENBLAS=ON -DCMAKE_BUILD_TYPE=Debug
make

# test Smurff:
wget http://homes.esat.kuleuven.be/~jsimm/chembl-IC50-346targets.mm
../_output/smurff --train chembl-IC50-346targets.mm

# run rests:
../_output/tests
```

## Installation using Conda
```bash
conda install -c vanderaa smurff 
```

# Contributors
- Jaak Simm (Macau C++ version, Cython wrapper, Macau MPI version, Tensor factorization)
- Tom Vander Aa (OpenMP optimized BPMF, Matrix Cofactorization and GFA, Code Reorg)
- Adam Arany (Probit noise model)
- Tom Haber (Original BPMF code)
- Andrei Gedich
- Ilya Pasechnikov

# Acknowledgements
This work was partly funded by the European projects ExCAPE (http://excape-h2020.eu) and
EXA2CT, and the Flemish Exaptation project.

