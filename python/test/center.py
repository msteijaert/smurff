#!/usr/bin/env python

import numpy as np
from scipy import sparse
import scipy.io as sio
import argparse 
import os
import os.path
import itertools
import matrix_io as mio

parser = argparse.ArgumentParser(description='Center sbm/sdm/ddm')
parser.add_argument('--mode',  metavar='MODE', dest='mode', action='store', help='global|rows|cols', default='global')
parser.add_argument('--train', help='Train input file', required = True)
parser.add_argument('--test', help='Test input file')
parser.add_argument('--row-features', nargs='+', help='Extra row features')
parser.add_argument('--col-features', nargs='+', help='Extra col features')
parser.add_argument('--output', help='Output directory', default='.')
args = parser.parse_args()
print(args)

if args.row_features: assert (args.mode in [ "cols", "none", "global" ] )
if args.col_features: assert (args.mode in [ "rows", "none", "global" ] )

def avg_sparse_cols(m):
    m = m.tocsc(copy=False) 
    sums = m.sum(axis=0).A1
    counts = np.diff(m.indptr)
    assert all(t > 0 for t in counts)
    mean = sums / counts
    sio.mmwrite("col_avg", np.expand_dims(mean, 0))
    return mean
 
def center_sparse_cols(m, mean):
    m = m.tocsr(copy=False) 
    m.data -= np.take(mean, m.indices)
    return m

def avg_sparse_rows(m):
    m = m.tocsr(copy=False) 
    sums = m.sum(axis=1).A1
    counts = np.diff(m.indptr)
    assert all(t > 0 for t in counts)
    mean = sums / counts
    return mean

def center_sparse_rows(m,mean):
    out = m.tocsc().copy() 
    out.data -= np.take(mean, out.indices)
    return out

def mean(m, mode):
    """compute mode-mean of matrix"""
    if (sparse.issparse(m)):
        if (args.mode == "rows"):     mean = avg_sparse_rows(m)
        elif (args.mode == "cols"):   mean = avg_sparse_cols(m)
        elif (args.mode == "global"): mean = np.mean(m.data)
        elif (args.mode == "none"):   mean = None
        else:
            raise ValueError("Unknown centering mode: %s" % ( args.mode ) )
    else:
        if (args.mode == "cols"):     mean = np.mean(m, 0)
        elif (args.mode == "rows"):   mean = np.mean(m, 1)
        elif (args.mode == "global"): mean = np.mean(m)
        elif (args.mode == "none"):   mean = None
        elif (args.mode != "none"):
            raise ValueError("Unknown centering mode: %s" % ( args.mode ) )

    if (args.mode == "cols"):     sio.mmwrite("mean_cols", np.expand_dims(mean, 0))
    elif (args.mode == "rows"):   sio.mmwrite("mean_rows", np.expand_dims(mean, 1))
    elif (args.mode == "global"): sio.mmwrite("mean_global", np.full([1,1], mean))

    return mean


def center(m, mode, mean):
    """center matrix m according to mode and computed mean"""
    if (sparse.issparse(m)):
        if (args.mode == "rows"):     m = center_sparse_rows(m,mean)
        elif (args.mode == "cols"):   m = center_sparse_cols(m,mean)
        elif (args.mode == "global"): m.data = m.data - mean
        elif (args.mode == "none"):   pass
        else:
            raise ValueError("Unknown centering mode: %s" % ( args.mode ) )
    else:
        if (args.mode == "cols"):     m = m - np.broadcast_to(np.expand_dims(mean, 0), m.shape)
        elif (args.mode == "rows"):   m = m - np.broadcast_to(np.expand_dims(mean, 1), m.shape)
        elif (args.mode == "global"): m = m - mean
        elif (args.mode == "none"):   pass
        elif (args.mode != "none"):
            raise ValueError("Unknown centering mode: %s" % ( args.mode ) )

    return m


train = mio.read_matrix(args.train)
test = mio.read_matrix(args.test)
assert train.shape == test.shape
mean_train = mean(train, args.mode)
centered_train = center(train, args.mode, mean_train)
centered_test = center(test, args.mode, mean_train)
mio.write_matrix(os.path.join(args.output, os.path.basename(args.train)), centered_train)
mio.write_matrix(os.path.join(args.output, os.path.basename(args.test)), centered_test)


features = []
if args.col_features: features += args.col_features 
if args.row_features: features += args.row_features 

for fname in features:
    m = mio.read_matrix(fname)
    if (args.mode == "global"): 
        mean_feat = mean_train
    else:
        mean_feat = mean(m, args.mode)
    centered_feat = center(m, args.mode, mean_feat)
    mio.write_matrix(os.path.join(args.output, os.path.basename(fname)), centered_feat)
