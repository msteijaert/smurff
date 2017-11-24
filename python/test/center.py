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
parser.add_argument('--mode',  metavar='MODE', dest='mode',  nargs=1, help='global|row|col', default='global')
parser.add_argument('infile', metavar='INFILE',  help='Input File')
parser.add_argument('outfile',  metavar='OUTFILE', help='Output File')
args = parser.parse_args()

m = mio.read_matrix(args.infile)

if (args.mode == "row"): 
    m = preprocessing.scale(m, axis = 0, with_std=False)
elif (args.mode == "col"):
    m = preprocessing.scale(m, axis = 1, with_std=False)
elif (args.mode == "global"): 
    d = m.data if (sparse.issparse(m)) else m
    d = d - np.mean(m)

mio.write_matrix(args.outfile, m)
