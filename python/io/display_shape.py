#!/usr/bin/env python
import matrix_io
import sys
import numpy as np

def shape_dense_float64(filename):
    with open(filename) as f:
        nrow = np.fromfile(f, dtype=np.int64, count=1)[0]
        ncol = np.fromfile(f, dtype=np.int64, count=1)[0]
        return (nrow, ncol)
 
for infile in sys.argv[1:]:
    print infile
    Y = shape_dense_float64(infile)
    print Y

