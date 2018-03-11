# SMURFF - Scalable Matrix Factorization Framework
[![Build Status](https://travis-ci.org/ExaScience/smurff.svg?branch=master)](https://travis-ci.org/ExaScience/smurff)
![Install with Conda](https://anaconda.org/vanderaa/smurff/badges/installer/conda.svg)


## What is Bayesian Matrix Factorization 

Matrix factorization is a common machine learning technique for recommender systems, like books for Amazon or movies for Netflix.


The idea of these methods is to approximate the user-movie rating matrix R as a
product of two low-rank matrices U and V (for the rest of the paper U refers to
the users matrix and V to the movie matrix) such that R ≈ U × V . In this way U
and V are constructed from the known ratings in R, which is usually very
sparsely filled. The recommendations can be made from the approximation U × V
which is dense. If M × N is the dimension of R then U and V will have
dimensions M × K and N × K.

Bayesian probabilistic matrix factorization (BPMF) has been proven to be more
robust to data-overfitting compared to non-Bayesian matrix factorization.

## What is SMURFF 
SMURFF is a highly optimized and parallelized framework for Bayesian Matrix and Tensors Factorization

SMURFF supports multiple matrix factorization methods: 
* [BPMF](https://www.cs.toronto.edu/~amnih/papers/bpmf.pdf), the basic version;
* [Macau](https://arxiv.org/abs/1509.04610), adding support for high-dimensional side information to the factorization;
* [GFA](https://arxiv.org/pdf/1411.5799.pdf), doing Group Factor Anaysis.

Macau and BPMF can also perform **tensor** factorization.

## Examples
For examples see [documentation](https://github.com/ExaScience/smurff/blob/master/python/jupyter-notebook/smurff.ipynb).

# Installation

Using [conda](http://anaconda.org):

```bash
conda install -c vanderaa smurff 
```
Compile from source code: see [INSTALL.md](docs/INSTALL.md)

## Contributors
- Jaak Simm (Macau C++ version, Cython wrapper, Macau MPI version, Tensor factorization)
- Tom Vander Aa (OpenMP optimized BPMF, Matrix Cofactorization and GFA, Code Reorg)
- Adam Arany (Probit noise model)
- Tom Haber (Original BPMF code)
- Andrei Gedich
- Ilya Pasechnikov

## Acknowledgements
This work was partly funded by the European projects ExCAPE (http://excape-h2020.eu) and
EXA2CT, and the Flemish Exaptation project.

