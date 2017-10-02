#!/usr/bin/env python
import matrix_io
import sys
import os
import pandas

for infile in sys.argv[1:]:
    outfile = os.path.splitext(infile)[0]+'.ddm'
    Y = pandas.read_csv(infile).as_matrix()
    matrix_io.write_dense_float64(outfile, Y)

