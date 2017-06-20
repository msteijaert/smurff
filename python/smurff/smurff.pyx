cimport cython
import numpy as np
cimport numpy as np
import scipy as sp
import timeit
import numbers
import pandas as pd
import signal

class PythonResult(object):
  def __init__(self):
    pass
  def __repr__(self):
    s = ("Matrix factorization results\n" +
         "Test RMSE:        %.4f\n" % self.rmse_avg +
         "Test AUC:         %.4f\n" % self.auc)
    return s

def version():
    cdef Config c
    return c.version

def bpmf(Y,
         Ytest      = None,
         num_latent = 10,
         precision  = 1.0,
         burnin     = 50,
         nsamples   = 400,
         **keywords):
    return macau(Y,
                 Ytest = Ytest,
                 num_latent = num_latent,
                 precision  = precision,
                 burnin     = burnin,
                 nsamples   = nsamples,
                 **keywords)

def remove_nan(Y):
    if not np.any(np.isnan(Y.data)):
        return Y
    idx = np.where(np.isnan(Y.data) == False)[0]
    return sp.sparse.coo_matrix( (Y.data[idx], (Y.row[idx], Y.col[idx])), shape = Y.shape )

cdef MatrixConfig prepare_sideinfo(in_matrix):
    if type(in_matrix) in [sp.sparse.coo.coo_matrix, sp.sparse.csr.csr_matrix, sp.sparse.csc.csc_matrix]:
        return prepare_sparse(in_matrix)
    else:
        return prepare_dense(in_matrix)
 
cdef MatrixConfig prepare_dense(X):
    cdef np.ndarray[np.double_t] vals = X.data.astype(np.double, copy=False)
    return MatrixConfig(X.shape[0], X.shape[1], & vals[0])

cdef MatrixConfig prepare_sparse(X):
    if type(X) not in [sp.sparse.coo.coo_matrix, sp.sparse.csr.csr_matrix, sp.sparse.csc.csc_matrix]:
        raise ValueError("Matrix must be either coo, csr or csc (from scipy.sparse)")
    X = X.tocoo(copy = False)
    X = remove_nan(X)
    cdef np.ndarray[int] irows = X.row.astype(np.int32, copy=False)
    cdef np.ndarray[int] icols = X.col.astype(np.int32, copy=False)
    cdef np.ndarray[np.double_t] vals = X.data.astype(np.double, copy=False)
    return MatrixConfig(X.shape[0], X.shape[1], irows.shape[0], & irows[0], & icols[0], & vals[0])

def macau(Y,
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

    cdef Config config

    # set config fields
    config.train = prepare_sparse(Y)
    if (Ytest): config.test = prepare_sparse(Ytest)
    config.verbose = verbose
    if (save_prefix): config.save_prefix = save_prefix
    config.nsamples = nsamples
    config.burnin = burnin

    # create session
    cdef PythonSession session
    session.setFromConfig(config)

    # only do one step at a time
    session.init()
    for i in range(nsamples + burnin):
            session.step()


    result = PythonResult()
    result.rmse = session.pred.rmse
    result.rmse_avg = session.pred.rmse_avg
    result.auc = session.pred.auc

    # df = pd.DataFrame({
    #   "row" : pd.Series(testdata[:,0], dtype='int'),
    #   "col" : pd.Series(testdata[:,1], dtype='int'),
    #   "y"   : pd.Series(testdata[:,2]),
    #   "y_pred" : pd.Series(yhat),
    #   "y_pred_std" : pd.Series(yhat_sd)
    # })

    return result

