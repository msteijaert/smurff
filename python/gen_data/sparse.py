import numpy as np
import scipy.sparse 

def read_sparse_float64(filename):
    with open(filename) as f:
        nrow = np.fromfile(f, dtype=np.int64, count=1)[0]
        ncol = np.fromfile(f, dtype=np.int64, count=1)[0]
        nnz  = np.fromfile(f, dtype=np.int64, count=1)[0]
        rows = np.fromfile(f, dtype=np.int32, count=nnz) - 1
        cols = np.fromfile(f, dtype=np.int32, count=nnz) - 1
        vals = np.fromfile(f, dtype=np.float64, count=nnz)
        return scipy.sparse.coo_matrix((vals, (rows, cols)), shape=[nrow, ncol])

def read_sparse_binary_matrix(filename):
    with open(filename) as f:
        nrow = np.fromfile(f, dtype=np.int64, count=1)[0]
        ncol = np.fromfile(f, dtype=np.int64, count=1)[0]
        nnz  = np.fromfile(f, dtype=np.int64, count=1)[0]
        rows = np.fromfile(f, dtype=np.int32, count=nnz) - 1
        cols = np.fromfile(f, dtype=np.int32, count=nnz) - 1
        return scipy.sparse.coo_matrix((np.ones(nnz), (rows, cols)), shape=[nrow, ncol])

def write_sparse_float64(filename, Y):
    with open(filename, 'w') as f:
        Y = Y.tocoo(copy = False)
        np.array(Y.shape[0]).astype(np.int64).tofile(f)
        np.array(Y.shape[1]).astype(np.int64).tofile(f)
        np.array(Y.nnz).astype(np.int64).tofile(f)
        (Y.row + 1).astype(np.int32, copy=False).tofile(f)
        (Y.col + 1).astype(np.int32, copy=False).tofile(f)
        Y.data.astype(np.float64, copy=False).tofile(f)

def write_sparse_binary_matrix(filename, Y):
    with open(filename, 'w') as f:
        Y = Y.tocoo(copy = False)
        np.array( Y.shape[0] ).astype(np.int64).tofile(f)
        np.array( Y.shape[1] ).astype(np.int64).tofile(f)
        np.array( Y.nnz ).astype(np.int64).tofile(f)
        (Y.row + 1).astype(np.int32, copy=False).tofile(f)
        (Y.col + 1).astype(np.int32, copy=False).tofile(f)

## example

# get the files from:
# 
# http://homes.esat.kuleuven.be/~jsimm/chembl-IC50-346targets.mm
# http://homes.esat.kuleuven.be/~jsimm/chembl-IC50-compound-feat.mm 
#
# import macau
# import scipy.io
# 
# ic50 = scipy.io.mmread("chembl-IC50-346targets.mm")
# ecfp = scipy.io.mmread("chembl-IC50-compound-feat.mm")
#
# import making_input_for_macau_mpi
# making_input_for_macau_mpi.write_sparse_binary_matrix("chembl-IC50-compound-feat.sbm", ecfp)
# making_input_for_macau_mpi.write_sparse_float64("chembl-IC50-346targets.sdm", ic50)
