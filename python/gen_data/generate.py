#!/usr/bin/env python

import numpy as np
from scipy import sparse
import scipy.io as sio
import argparse 
import os
import matrix_io as mio

#parser = argparse.ArgumentParser(description='SMURFF tests')
#parser.add_argument('--envdir',  metavar='DIR', dest='envdir',  nargs=1, help='Env dir', default='conda_envs')
#parser.add_argument('--data', metavar='DIR', dest='datadir', nargs=1, help='Data dir', default='data')
#parser.add_argument('--outdir',  metavar='DIR', dest='outdir', nargs=1, help='Output dir',
#        default = 'work/' + datetime.datetime.today().strftime("%Y%m%d-%H%M%S"))
#
#args = parser.parse_args()

# product of two gaussian low-rank matrices + noise
def normal_dense(N, D, K):
    X = np.random.normal(size=(N,K))
    W = np.random.normal(size=(D,K))
    return np.dot(X, W.transpose()) + np.random.normal(size=(N,D))

# product of two low-rank 'ones' matrices
def ones_dense(N, D, K):
    X = np.ones(size=(N,K))
    W = np.ones(size=(D,K))
    return np.dot(X,W.transpose())

# dense -> sparse
#  or
# sparse -> even sparser
def sparsify(A, density):
    num = int(A.size * density)
    idx = np.random.choice(A.size, num, replace=False)
    (I, J, V) = sparse.find(sparse.coo_matrix(A))
    return sparse.coo_matrix((V[idx], (I[idx], J[idx])), shape = A.shape)

def write_matrix(filename, A):
    if sparse.issparse(A):
        mio.write_sparse_float64(filename + ".sdm", A)
    else:
        mio.write_dense_float64(filename + ".ddm", A)

    sio.mmwrite(filename, A)


def write_features(basename, features)
    no = 0
    for F in features:
        write_matrix("%s_%s" % (basename, no), F)
        no += 1 

def write_data(dirname, train, row_features = [], col_features = []):
    os.makedirs(dirname)
    os.chdir(dirname)
    write_matrix("train", train)
    test = sparsify(train, 0.2)
    write_matrix("test", test)
    write_feat("col", col_features)
    write_feat("row", row_features)
    os.chdir("..")

if __name__ == "__main__":
    write_data("normal_dense_200_100_4", normal_dense(200,100,4))
    write_data("normal_sparse_2000_100_4", sparsify(normal_dense(2000,100,4), 0.2))


