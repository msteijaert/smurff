#!/usr/bin/env python3

import matrix_io as mio
import sys

for f in sys.argv[1:]:
    m = mio.read_matrix(f)
    try:
        print(f, ":", m.shape, m.nnz)
    except AttributeError:
        print(f, ":", m.shape)
