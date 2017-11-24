#!/usr/bin/env python

import numpy as np
from scipy import sparse
import scipy.io as sio
import argparse 
import os
import os.path
import itertools
import matrix_io as mio
from sklearn import preprocessing

parser = argparse.ArgumentParser(description='Center sbm/sdm/ddm')
parser.add_argument('--mode',  metavar='MODE', dest='mode', action='store', help='global|rows|cols', default='global')
parser.add_argument('train', help='Train input file')
parser.add_argument('test', help='Test input file')
args = parser.parse_args()

print(args)
train = mio.read_matrix(args.train)
test = mio.read_matrix(args.test)

def avg_sparse_cols(m):
    m = m.tocsc(copy=False) 
    sums = m.sum(axis=0).A1
    counts = np.diff(m.indptr)
    avg = sums / (counts + 0.000001)
    avg = np.expand_dims(avg, 0)
    return np.broadcast_to(avg, m.shape)
 
def center_sparse_cols(m, avg):
    m = m.tocsc(copy=False) 
    m.data -= np.take(avg, m.indices)
    return m

def avg_sparse_rows(m):
    m = m.tocsr(copy=False) 
    sums = m.sum(axis=1).A1
    counts = np.diff(m.indptr)
    avg = sums / (counts + 0.000001)
    avg = np.expand_dims(avg, 1)
    return np.broadcast_to(avg, m.shape)

def center_sparse_rows(m,avg):
    m = m.tocsr(copy=False) 
    m.data -= np.take(avg, m.indices)
    return m

if (sparse.issparse(train)):
    if (args.mode == "rows"): 
        a = avg_sparse_rows(train)
        train = center_sparse_rows(train,a)
        test = center_sparse_rows(test,a)
    elif (args.mode == "cols"):
        a = avg_sparse_cols(train)
        train = center_sparse_cols(train,a)
        test = center_sparse_cols(test,a)
    elif (args.mode == "global"): 
        a = np.mean(train)
        train.data = train.data - a
        test.data = test.data - a
    elif (args.mode != "none"):
        raise ValueError("Unknown centering mode: %s" % ( args.mode ) )
else:
    if (args.mode == "rows"): 
        a = np.mean(train, 0)
        train = train - np_broadcast_to(train.shape)
        test = center_sparse_rows(test,a)
    elif (args.mode == "cols"):
        a = np.mean(train, 1)
        train = train - np_broadcast_to(train.shape)
        test = center_sparse_cols(test,a)
    elif (args.mode == "global"): 
        a = np.mean(train)
        train = train - np_broadcast_to(train.shape)
        test.data = test.data - a
    elif (args.mode != "none"):
        raise ValueError("Unknown centering mode: %s" % ( args.mode ) )

sio.mmwrite("centered_train", train)
sio.mmwrite("centered_test", test)

