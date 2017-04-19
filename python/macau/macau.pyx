cimport cython
import numpy as np
cimport numpy as np
import scipy as sp
import timeit
import numbers
import pandas as pd
import signal

class MacauResult(object):
  def __init__(self):
    pass
  def __repr__(self):
    s = ("Matrix factorization results\n" +
         "Test RMSE:        %.4f\n" % self.rmse_test +
         "Matrix size:      [%d x %d]\n" % (self.Yshape[0], self.Yshape[1]) +
         "Number of train:  %d\n" % self.ntrain +
         "Number of test:   %d\n" % self.ntest  +
         "To see predictions on test set see '.prediction' field.")
    return s

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
        return prepare_sparse_sideinfo(in_matrix)
    else:
        return prepare_dense_sideinfo(in_matrix)
 
cdef MatrixConfig prepare_dense_sideinfo(X):
    cdef np.ndarray[np.double_t] vals = X.data.astype(np.double, copy=False)
    return MatrixConfig(X.shape[0], X.shape[1], & vals[0])

cdef MatrixConfig prepare_sparse_sideinfo(X):
    X = X.to_coo(copy = False)
    cdef np.ndarray[int] irows = X.row.astype(np.int32, copy=False)
    cdef np.ndarray[int] icols = X.col.astype(np.int32, copy=False)
    cdef np.ndarray[np.double_t] vals = X.data.astype(np.double, copy=False)
    return MatrixConfig(X.shape[0], X.shape[1], irows.shape[0], & irows[0], & icols[0], & vals[0])

def prepare_sparse(M):
    if type(M) not in [sp.sparse.coo.coo_matrix, sp.sparse.csr.csr_matrix, sp.sparse.csc.csc_matrix]:
        raise ValueError("Matrix must be either coo, csr or csc (from scipy.sparse)")
    M = M.tocoo(copy = False)
    M = remove_nan(M)
    return M

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

    config = MacauConfig 

    Y = prepare_sparse(Y)
    Ytest = prepare_sparse(Ytest)

    cdef PythonSession session
    session.step()

    #cdef int D = np.int32(num_latent)
    # cdef unique_ptr[ILatentPrior] prior_u
    # cdef unique_ptr[ILatentPrior] prior_v
    # if univariate:
    #     prior_u = unique_ptr[ILatentPrior](make_one_prior(side[0], D, lambda_beta))
    #     prior_v = unique_ptr[ILatentPrior](make_one_prior(side[1], D, lambda_beta))
    # else:
    #     prior_u = unique_ptr[ILatentPrior](make_prior(side[0], D, 10000, lambda_beta, tol))
    #     prior_v = unique_ptr[ILatentPrior](make_prior(side[1], D, 10000, lambda_beta, tol))

    #cdef Macau *macau = new Macau(D)
    #macau.addPrior(prior_u)
    #macau.addPrior(prior_v)
    #macau.setRelationData(&irows[0], &icols[0], &ivals[0], irows.shape[0], Y.shape[0], Y.shape[1]);
    #macau.setSamples(np.int32(burnin), np.int32(nsamples))
    #macau.setVerbose(verbose)

    #if isinstance(precision, str):
    #  if precision == "adaptive" or precision == "sample":
    #    macau.setAdaptivePrecision(np.float64(1.0), np.float64(sn_max))
    #  elif precision == "probit":
    #    macau.setProbit()
    #  else:
    #    raise ValueError("Parameter 'precision' has to be either a number or \"adaptive\" for adaptive precision, or \"probit\" for binary matrices.")
    #else:
    #  macau.setPrecision(np.float64(precision))

    #cdef np.ndarray[int] trows, tcols
    #cdef np.ndarray[np.double_t] tvals

    #if Ytest is not None:
    #    trows = Ytest.row.astype(np.int32, copy=False)
    #    tcols = Ytest.col.astype(np.int32, copy=False)
    #    tvals = Ytest.data.astype(np.double, copy=False)
    #    macau.setRelationDataTest(&trows[0], &tcols[0], &tvals[0], trows.shape[0], Y.shape[0], Y.shape[1])

    #if save_prefix is None:
    #    macau.setSaveModel(0)
    #else:
    #    if type(save_prefix) != str:
    #        raise ValueError("Parameter 'save_prefix' has to be a string (str) or None.")
    #    macau.setSaveModel(1)
    #    macau.setSavePrefix(save_prefix)

    #macau.run()
    ## restoring Python default signal handler
    # signal.signal(signal.SIGINT, signal.default_int_handler)

    # cdef VectorXd yhat_raw     = macau.getPredictions()
    # cdef VectorXd yhat_sd_raw  = macau.getStds()
    # cdef MatrixXd testdata_raw = macau.getTestData()

    # cdef np.ndarray[np.double_t] yhat    = vecview( & yhat_raw ).copy()
    # cdef np.ndarray[np.double_t] yhat_sd = vecview( & yhat_sd_raw ).copy()
    # cdef np.ndarray[np.double_t, ndim=2] testdata = matview( & testdata_raw ).copy()

    # df = pd.DataFrame({
    #   "row" : pd.Series(testdata[:,0], dtype='int'),
    #   "col" : pd.Series(testdata[:,1], dtype='int'),
    #   "y"   : pd.Series(testdata[:,2]),
    #   "y_pred" : pd.Series(yhat),
    #   "y_pred_std" : pd.Series(yhat_sd)
    # })

    result = MacauResult()
    result.rmse_test  = macau.getRmseTest()
    result.Yshape     = Y.shape
    result.ntrain     = Y.nnz
    result.ntest      = Ytest.nnz if Ytest is not None else 0
    #result.prediction = pd.DataFrame(df)

    #del macau

    return result

