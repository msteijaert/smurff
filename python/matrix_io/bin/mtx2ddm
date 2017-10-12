#!/usr/bin/env python
from scipy.io import mmread
import matrix_io
import sys
import os

for infile in sys.argv[1:]:
    outfile = os.path.splitext(infile)[0]+'.ddm'
    Y = mmread(infile).todense()
    matrix_io.write_dense_float64(outfile, Y)

