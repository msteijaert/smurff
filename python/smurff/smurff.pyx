from libc.stdint cimport *
from libcpp.memory cimport shared_ptr, make_shared
from libcpp.vector cimport vector

from Config cimport Config, stringToPriorType, stringToModelInitType
from ISession cimport ISession
from NoiseConfig cimport NoiseConfig
from MatrixConfig cimport MatrixConfig
from SessionFactory cimport SessionFactory

cimport numpy as np
import  numpy as np
import  scipy as sp

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
           Ytest,
           row_features   = [],
           col_features   = [],
           row_prior      = None,
           col_prior      = None,
           lambda_beta    = 5.0,
           num_latent     = 10,
           precision      = 1.0,
           adaptive       = None,
           burnin         = 50,
           nsamples       = 400,
           tol            = 1e-6,
           direct         = True,
           seed           = None,
           threshold      = None,
           verbose        = True,
           quite          = False,
           init_model     = None,
           save_prefix    = None,
           save_suffix    = None,
           save_freq      = None,
           restore_prefix = None,
           restore_suffix = None,
           csv_status     = None):

    # Create and initialize smurff-cpp Config instance
    cdef Config config

    config.train = prepare_sparse(Y).get()[0]
    config.test  = prepare_sparse(Ytest).get()[0]

    for rf in row_features:
        config.row_features.push_back(prepare_sparse(rf).get()[0])

    for cf in col_features:
        config.col_features.push_back(prepare_sparse(cf).get()[0])

    if row_prior:
        config.row_prior_type = stringToPriorType(row_prior)

    if col_prior:
        config.col_prior_type = stringToPriorType(col_prior)

    config.lambda_beta = lambda_beta
    config.num_latent  = num_latent
    # config.precision   = ???
    # config.adaptive    = ???
    config.burnin      = burnin
    config.nsamples    = nsamples
    config.tol         = tol
    config.direct      = direct

    if seed:
        config.random_seed_set = True
        config.random_seed = seed

    if threshold:
        config.threshold = threshold
        config.classify = True

    config.verbose = verbose
    if quite:
        config.verbose = False

    if init_model:
        config.model_init_type = stringToModelInitType(init_model)

    if save_prefix:
        config.setSavePrefix(save_prefix)

    if save_suffix:
        config.save_suffix = save_suffix

    if save_freq:
        config.save_freq = save_freq

    if restore_prefix:
        config.restore_prefix = restore_prefix

    if restore_suffix:
        config.restore_suffix = restore_suffix

    if csv_status:
        config.csv_status = csv_status

    # Create and run session
    cdef shared_ptr[ISession] session = SessionFactory.create_py_session(config)
    session.get().run()

    # Get result from session and construct scipy matrix
    cdef shared_ptr[MatrixConfig] result = make_shared[MatrixConfig](session.get().getResult())

    cdef shared_ptr[vector[uint32_t]] result_rows_ptr = result.get().getRowsPtr()
    cdef uint32_t[:] result_rows_view = <uint32_t[:result_rows_ptr.get().size()]>result_rows_ptr.get().data()
    result_rows = np.array(result_rows_view, copy=False)

    cdef shared_ptr[vector[uint32_t]] result_cols_ptr = result.get().getColsPtr()
    cdef uint32_t[:] result_cols_view = <uint32_t[:result_cols_ptr.get().size()]>result_cols_ptr.get().data()
    result_cols = np.array(result_cols_view, copy=False)

    cdef shared_ptr[vector[double]] result_vals_ptr = result.get().getValuesPtr()
    cdef double[:] result_vals_view = <double[:result_vals_ptr.get().size()]>result_vals_ptr.get().data()
    result_vals = np.array(result_vals_view, copy=False)

    result_sparse_matrix = sp.sparse.coo_matrix((result_vals, (result_rows, result_cols)), shape=(result.get().getNRow(), result.get().getNCol()))
    return result_sparse_matrix