#!/usr/bin/env python

import numpy as np
from scipy import sparse
import scipy.io as sio
import argparse 
import os
import os.path
import itertools
import matrix_io as mio

parser = argparse.ArgumentParser(description='Remove duplicates from sbm/sdm')
parser.add_argument('input', help='Input file')
parser.add_argument('output', help='Output file')
args = parser.parse_args()

print(args)
n = mio.read_sparse_float64(args.input)
assert sparse.issparse(n)

# converting to dok matrix removes dupes
mio.write_sparse_float64(args.output, n.todok())

