#!/usr/bin/env python

import numpy as np
from scipy import sparse
import scipy.io as sio
import argparse
import os
import itertools
import matrix_io as mio
from sklearn import preprocessing

def write_dense_float64_col_major(filename, Y):
    with open(filename, 'wb') as f:
        np.array(Y.shape[0]).astype(np.int64).tofile(f)
        np.array(Y.shape[1]).astype(np.int64).tofile(f)
        f.write(Y.astype(np.float64).tobytes(order='F'))

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

def col_rep(shape, K):
    W = np.arange(shape[1]).reshape(1, shape[1])
    return np.repeat(W, shape[0], 0)

# dense -> sparse
#  or
# sparse -> even sparser
def sparsify(A, density):
    if sparse.issparse(A):
        (I, J, V) = sparse.find(A)
    else:
        V = A.reshape(A.size)
        (I, J) = np.indices(A.shape)
        I = I.reshape(A.size)
        J = J.reshape(A.size)

    size = V.size
    num = int(size * density) 
    idx = np.random.choice(size, num, replace=False)

    return sparse.coo_matrix((V[idx], (I[idx], J[idx])), shape = A.shape)

def gen_matrix(shape, K, func = "normal", density = 1.0 ):
    func_dict = {
            "normal": normal_dense,
            "ones":   ones_dense,
            "col":    col_rep,
    }

    m = func_dict[func] (shape,K)

    if density < 1.0:
        m = sparsify(m, density)

    return m

def write_matrix(filename, A):
    if sparse.issparse(A):
        mio.write_sparse_float64(filename + ".sdm", A)
    else:
        # TRANSPOSE BECAUSE OF INTERNAL REPRESENTATION OF DENSE
        # mio.write_dense_float64(filename + ".ddm", A.transpose())
        # mio.write_dense_float64(filename + ".ddm", A)
        write_dense_float64_col_major(filename + ".ddm", A)

    sio.mmwrite(filename, A)

def write_feat(base, features):
    for (indx,F) in enumerate(features):
        write_matrix("feat_%d_%d" % (base, indx), F)

def write_test_data(dirname, test):
    os.chdir(dirname)
    write_matrix("test", test)
    os.chdir("..")

def write_train_data(dirname, train, features = ([],[])):
    os.makedirs(dirname)
    os.chdir(dirname)
    write_matrix("train", train)
    for (indx,feat) in enumerate(features):
        write_feat(indx, feat)
    os.chdir("..")

def gen_test_and_write(m, shape, K,func,density, row_split = 1, col_split = 1, center_list=["none"]):
    # split rows and cols
    rows_blocked = np.array_split(m, row_split, axis=0)
    blocks = [ np.array_split(b, col_split, axis=1) for b in rows_blocked]
    m = blocks[0][0]
    col_feat = [b[0] for b in blocks[1:]]
    row_feat = blocks[0][1:]

    assert len(col_feat) == row_split - 1
    assert len(row_feat) == col_split - 1

    for r in row_feat: assert r.shape[0] == m.shape[0]
    for r in col_feat: assert r.shape[1] == m.shape[1]

    test = sparsify(m, 0.2)

    for center in center_list:
        if (func == "ones" and center != "none"):
            continue

        shape_str = "_".join(map(str,shape))
        dirname = "%s_%s_%d_%d_%d_%d_%s" % (func, shape_str, K, int(density * 100), row_split, col_split, center)

        print("%s..." % dirname)
        write_test_data(dirname, test)

def gen_train_and_write(m, shape, K,func,density, row_split = 1, col_split = 1, center = "none"):
    if (func == "ones" and center != "none"):
        return
    
    shape_str = "_".join(map(str,shape))
    dirname = "%s_%s_%d_%d_%d_%d_%s" % (func, shape_str, K, int(density * 100), row_split, col_split, center)

    if os.path.exists(dirname):
        print("Already exists: %s. Skipping" % dirname)
        return

    print("%s..." % dirname)

    # split rows and cols
    rows_blocked = np.array_split(m, row_split, axis=0)
    blocks = [ np.array_split(b, col_split, axis=1) for b in rows_blocked]
    m = blocks[0][0]
    col_feat = [b[0] for b in blocks[1:]]
    row_feat = blocks[0][1:]

    assert len(col_feat) == row_split - 1
    assert len(row_feat) == col_split - 1

    for r in row_feat: assert r.shape[0] == m.shape[0]
    for r in col_feat: assert r.shape[1] == m.shape[1]

    # PAY ATTENTION TO AXIS ORDER
    if (center == "row"):
        m = preprocessing.scale(m, axis = 0, with_std=False)
    elif (center == "col"):
        m = preprocessing.scale(m, axis = 1, with_std=False)
    elif (center == "global"):
        m = m - np.mean(m)

    for i in range(len(row_feat)):
        if (center == "row"):
            row_feat[i] = preprocessing.scale(row_feat[i], axis = 0, with_std=False)
        elif (center == "col"):
            row_feat[i] = preprocessing.scale(row_feat[i], axis = 1, with_std=False)
        elif (center == "global"):
            row_feat[i] = row_feat[i] - np.mean(row_feat[i])

    for i in range(len(col_feat)):
        if (center == "row"):
            col_feat[i] = preprocessing.scale(col_feat[i], axis = 0, with_std=False)
        elif (center == "col"):
            col_feat[i] = preprocessing.scale(col_feat[i], axis = 1, with_std=False)
        elif (center == "global"):
            col_feat[i] = col_feat[i] - np.mean(col_feat[i])

    write_train_data(dirname, m, (row_feat, col_feat))


if __name__ == "__main__":
    shape = [2000,100]
    #shape = [40,30]
    num_latent = 4
    # for density in (1, .2):
    for density in (1,):
        for func in ("normal", "ones"):
            # CALL GEN MATRIX ONLY ONCE
            m = gen_matrix(shape,num_latent,func)
            for row_split in (1,2,3):
                for col_split in (1,2,3,):
                    for center in ("none", "global", "row", "col"):
                        gen_train_and_write(m,shape,num_latent,func,density, row_split, col_split, center)
                    # SPARSIFY SHOULD BE CALLED ONLY ONCE
                    gen_test_and_write(m,shape,num_latent,func,density, row_split, col_split, ("none", "global", "row", "col"))