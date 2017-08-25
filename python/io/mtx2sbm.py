#!/usr/bin/env python
from scipy.io import mmread
import matrix_io
import sys
import os

for infile in sys.argv[1:]:
    outfile = os.path.splitext(infile)[0]+'.sbm'
    Y = mmread(infile)
    matrix_io.write_sparse_binary_matrix(outfile, Y)

