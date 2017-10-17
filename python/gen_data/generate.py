#!/usr/bin/env python

import numpy as np
from scipy import sparse
import scipy.io as sio
import argparse 
import os
import itertools
import matrix_io as mio

#parser = argparse.ArgumentParser(description='SMURFF tests')
#parser.add_argument('--envdir',  metavar='DIR', dest='envdir',  nargs=1, help='Env dir', default='conda_envs')
#parser.add_argument('--data', metavar='DIR', dest='datadir', nargs=1, help='Data dir', default='data')
#parser.add_argument('--outdir',  metavar='DIR', dest='outdir', nargs=1, help='Output dir',
#        default = 'work/' + datetime.datetime.today().strftime("%Y%m%d-%H%M%S"))
#
#args = parser.parse_args()

# product of two gaussian low-rank matrices + noise
def normal_dense(shape, K):
    X = np.random.normal(size=(shape[0],K))
    W = np.random.normal(size=(shape[1],K))
    return np.dot(X, W.transpose()) + np.random.normal(size=shape)

# product of two low-rank 'ones' matrices
def ones_dense(shape, K):
    X = np.ones((shape[0],K))
    W = np.ones((shape[1],K))
    return np.dot(X,W.transpose())

# dense -> sparse
#  or
# sparse -> even sparser
def sparsify(A, density):
    num = int(A.size * density)
    idx = np.random.choice(A.size, num, replace=False)
    (I, J, V) = sparse.find(sparse.coo_matrix(A))
    return sparse.coo_matrix((V[idx], (I[idx], J[idx])), shape = A.shape)

def gen_matrix(shape, K, func = "normal", density = 1.0 ):
    if func == "normal":
        m = normal_dense(shape,K)
    elif func == "ones":
        m = ones_dense(shape,K)
    else:
        assert False

    if density < 1.0:
        m = sparsify(m, density)

    return m

def write_matrix(filename, A):
    if sparse.issparse(A):
        mio.write_sparse_float64(filename + ".sdm", A)
    else:
        mio.write_dense_float64(filename + ".ddm", A)

    sio.mmwrite(filename, A)

def write_feat(base, features):
    for (indx,F) in enumerate(features):
        write_matrix("feat_%d_%d" % (base, indx), F)

def write_data(dirname, train, features = ([],[])):
    os.makedirs(dirname)
    os.chdir(dirname)
    write_matrix("train", train)
    test = sparsify(train, 0.2)
    write_matrix("test", test)
    for (indx,feat) in enumerate(features):
        write_feat(indx, feat)
    os.chdir("..")

def gen_and_write(shape, K,func,density, row_split = 1, col_split = 1):
    m = gen_matrix(shape,K,func);

    rows_blocked = np.array_split(m, row_split, axis=0)
    blocks = [ np.array_split(b, col_split, axis=1) for b in rows_blocked]
    m = blocks[0][0]
    row_feat = [b[0] for b in blocks[1:]]
    col_feat = blocks[0][1:]
    assert len(col_feat) == col_split - 1
    assert len(row_feat) == row_split - 1

    if density < 1.0:
        m = sparsify(m, density)

    shape_str = "_".join(map(str,shape))
    dirname = "%s_%s_%d_%d_%d_%d" % (func, shape_str, K, int(density * 100), row_split, col_split)
    write_data(dirname, m, (row_feat, col_feat))

if __name__ == "__main__":
    shape = [2000,100]
    num_latent = 4
    for density in (1, .2):
        for func in ("normal", "ones"):
            for row_split in (1,2,3):
                for col_split in (1,2,3,):
                    gen_and_write(shape,num_latent,func,density, row_split, col_split)
