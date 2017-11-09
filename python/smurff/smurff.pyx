from libc.stdint cimport *
from libcpp.memory cimport shared_ptr, make_shared
from libcpp.vector cimport vector

from Config cimport Config
from Session cimport Session
from NoiseConfig cimport NoiseConfig
from MatrixConfig cimport MatrixConfig
from SessionFactory cimport SessionFactory

cimport numpy as np

import numpy as np
import scipy as sp

def remove_nan(Y):
    if not np.any(np.isnan(Y.data)):
        return Y
        idx = np.where(np.isnan(Y.data) == False)[0]
        return sp.sparse.coo_matrix( (Y.data[idx], (Y.row[idx], Y.col[idx])), shape = Y.shape )

cdef shared_ptr[MatrixConfig] prepare_sparse(X):
    if type(X) not in [sp.sparse.coo.coo_matrix, sp.sparse.csr.csr_matrix, sp.sparse.csc.csc_matrix]:
        raise ValueError("Matrix must be either coo, csr or csc (from scipy.sparse)")
    X = X.tocoo(copy = False)
    X = remove_nan(X)
    cdef np.ndarray[uint32_t] irows = X.row.astype(np.uint32, copy=False)
    cdef np.ndarray[uint32_t] icols = X.col.astype(np.uint32, copy=False)
    cdef np.ndarray[double] vals = X.data.astype(np.double, copy=False)

    # Get begin and end pointers from numpy arrays
    cdef uint32_t* irows_begin = &irows[0]
    cdef uint32_t* irows_end = irows_begin + irows.shape[0]
    cdef uint32_t* icols_begin = &icols[0]
    cdef uint32_t* icols_end = icols_begin + icols.shape[0]
    cdef double* vals_begin = &vals[0]
    cdef double* vals_end = vals_begin + vals.shape[0]

    # Create vectors from pointers
    cdef vector[uint32_t] rows_vector
    rows_vector.assign(irows_begin, irows_end)
    cdef vector[uint32_t] cols_vector
    cols_vector.assign(icols_begin, icols_end)
    cdef vector[double] vals_vector
    vals_vector.assign(vals_begin, vals_end)

    # Kind of a hack, because to return a MatrixConfig it is needed to have a default constructor
    # So shared_ptr is used to overcome this cython limitation
    return make_shared[MatrixConfig](<uint64_t>(X.shape[0]), <uint64_t>(X.shape[1]), rows_vector, cols_vector, vals_vector, NoiseConfig())

def smurff(Y,
           Ytest        = None,
           row_features = [],
           col_features = [],
           row_prior    = None,
           col_prior    = None,
           lambda_beta  = 5.0,
           num_latent   = 10,
           precision    = 1.0,
           burnin       = 50,
           nsamples     = 400,
           tol          = 1e-6,
           sn_max       = 10.0,
           save_prefix  = None,
           verbose      = True):

    # Create and initialize smurff-cpp Config instance
    cdef Config config
    config.train = prepare_sparse(Y).get()[0]
    if (Ytest):
        config.test = prepare_sparse(Ytest).get()[0]
    config.verbose = verbose
    if (save_prefix):
        config.setSavePrefix(save_prefix)
    config.nsamples = nsamples
    config.burnin = burnin
    config.num_latent = num_latent

    # Create and run session
    cdef shared_ptr[Session] session = SessionFactory.create_py_session(config)
    session.get().run()
