# Macau-cpp - Bayesian Factorization with Side Information
Highly optimized and parallelized methods for Bayesian Factorization, including BPMF and Macau. The package uses optimized OpenMP/C++ code with a Cython wrapper to factorize large scale matrices. Macau method provides also the ability to incorporate high-dimensional side information to the factorization.

# Examples
For examples see [documentation](http://macau.readthedocs.io/en/latest/source/examples.html).

# Installation
Currently there is only C++ version of smurff available. Python one is comming soon.

## Source installation on Ubuntu
```bash
# install dependencies:
sudo apt-get install libopenblas-dev autoconf gfortran

# checkout and install Smurff
git clone https://github.com/ExaScience/smurff.git
cd smurff
git checkout smurff
git submodules init
git submodules install
cd lib/smurff-lib/makefiles/linux
make

# test Smurff
wget http://homes.esat.kuleuven.be/~jsimm/chembl-IC50-346targets.mm
./smurff --train chembl-IC50-346targets.mm
```

# Contributors
- Jaak Simm (Macau C++ version, Cython wrapper, Macau MPI version)
- Adam Arany (Probit noise model)
- Tom Vander Aa (OpenMP optimized BPMF)
- Tom Haber (Original BPMF code)
