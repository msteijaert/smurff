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
    X = np.ones((N,K))
    W = np.ones((D,K))
    return np.dot(X,W.transpose())

# dense -> sparse
#  or
# sparse -> even sparser
def sparsify(A, density):
    num = int(A.size * density)
    idx = np.random.choice(A.size, num, replace=False)
    (I, J, V) = sparse.find(sparse.coo_matrix(A))
    return sparse.coo_matrix((V[idx], (I[idx], J[idx])), shape = A.shape)


def gen_matrix(N, D, K, func = "normal", density = 1.0 ):
    if func == "normal":
        m = normal_dense(N,D,K)
    elif func == "ones":
        m = ones_dense(N,D,K)
    else:
        assert False

    if density < 1.0:
        m = sparisify(m, density)

    return m

def write_matrix(filename, A):
    if sparse.issparse(A):
        mio.write_sparse_float64(filename + ".sdm", A)
    else:
        mio.write_dense_float64(filename + ".ddm", A)

    sio.mmwrite(filename, A)

def write_feat(basename, features):
    no = 0
    for F in features:
        write_matrix("%s_%d" % (basename, no), F)
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

def gen_and_write(N, D, K,func,density):
    m = gen_matrix(N,D,K,func,density);
    write_data("%s_%d_%d_%d_%d" % (func, N, D, K, int(density * 100)), m)

if __name__ == "__main__":
    for density in (1, .2):
        for func in ("normal", "ones"):
            gen_and_write(200,100,4,func,density)

    # 1 set of row-featueres


    # 2 sets of row-features


