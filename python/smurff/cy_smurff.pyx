##
#
# Cython file that bridges between the smurff python code and the C++ code
# 
# This file will be compiled via CMake
#


from libc.stdint cimport *
from libcpp.memory cimport shared_ptr, make_shared
from libcpp.vector cimport vector

from Config cimport Config, PriorTypes, stringToPriorType, stringToModelInitType
from ISession cimport ISession
from NoiseConfig cimport *
from MatrixConfig cimport MatrixConfig
from TensorConfig cimport TensorConfig
from SessionFactory cimport SessionFactory
from PVec cimport PVec

cimport numpy as np
import  numpy as np
import  scipy as sp
import pandas as pd
import scipy.sparse
import numbers

from .prepare import make_train_test, make_train_test_pd
from . import Result, ResultItem

DENSE_MATRIX_TYPES  = [np.ndarray, np.matrix]
SPARSE_MATRIX_TYPES = [sp.sparse.coo.coo_matrix, sp.sparse.csr.csr_matrix, sp.sparse.csc.csc_matrix]

DENSE_TENSOR_TYPES  = [np.ndarray]
SPARSE_TENSOR_TYPES = [pd.DataFrame]

def remove_nan(Y):
    if not np.any(np.isnan(Y.data)):
        return Y
    idx = np.where(np.isnan(Y.data) == False)[0]
    return sp.sparse.coo_matrix( (Y.data[idx], (Y.row[idx], Y.col[idx])), shape = Y.shape )

cdef MatrixConfig* prepare_sparse_matrix(X, is_scarse):
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
    cdef vector[uint32_t]* rows_vector_ptr = new vector[uint32_t]()
    rows_vector_ptr.assign(irows_begin, irows_end)
    cdef vector[uint32_t]* cols_vector_ptr = new vector[uint32_t]()
    cols_vector_ptr.assign(icols_begin, icols_end)
    cdef vector[double]* vals_vector_ptr = new vector[double]()
    vals_vector_ptr.assign(vals_begin, vals_end)

    cdef shared_ptr[vector[uint32_t]] rows_vector_shared_ptr = shared_ptr[vector[uint32_t]](rows_vector_ptr)
    cdef shared_ptr[vector[uint32_t]] cols_vector_shared_ptr = shared_ptr[vector[uint32_t]](cols_vector_ptr)
    cdef shared_ptr[vector[double]] vals_vector_shared_ptr = shared_ptr[vector[double]](vals_vector_ptr)

    cdef MatrixConfig* matrix_config_ptr = new MatrixConfig(<uint64_t>(X.shape[0]), <uint64_t>(X.shape[1]), rows_vector_shared_ptr, cols_vector_shared_ptr, vals_vector_shared_ptr, NoiseConfig(), is_scarse)
    return matrix_config_ptr

cdef MatrixConfig* prepare_dense_matrix(X):
    cdef np.ndarray[np.double_t] vals = np.squeeze(np.asarray(X.flatten(order='F')))
    cdef double* vals_begin = &vals[0]
    cdef double* vals_end = vals_begin + vals.shape[0]
    cdef vector[double]* vals_vector_ptr = new vector[double]()
    vals_vector_ptr.assign(vals_begin, vals_end)
    cdef shared_ptr[vector[double]] vals_vector_shared_ptr = shared_ptr[vector[double]](vals_vector_ptr)
    cdef MatrixConfig* matrix_config_ptr = new MatrixConfig(<uint64_t>(X.shape[0]), <uint64_t>(X.shape[1]), vals_vector_shared_ptr, NoiseConfig())
    return matrix_config_ptr

cdef MatrixConfig* prepare_sideinfo(in_matrix):
    if type(in_matrix) in [sp.sparse.coo.coo_matrix, sp.sparse.csr.csr_matrix, sp.sparse.csc.csc_matrix]:
        return prepare_sparse_matrix(in_matrix, False)
    else:
        return prepare_dense_matrix(in_matrix)

cdef TensorConfig* prepare_dense_tensor(tensor):
    if type(tensor) not in DENSE_TENSOR_TYPES:
        error_msg = "Unsupported dense tensor data type: {}".format(tensor)
        raise ValueError(error_msg)
    raise NotImplementedError()

cdef TensorConfig* prepare_sparse_tensor(tensor, shape, is_scarse):
    if type(tensor) not in SPARSE_TENSOR_TYPES:
        error_msg = "Unsupported sparse tensor data type: {}".format(tensor)
        raise ValueError(error_msg)

    cdef vector[uint64_t] cpp_dims_vector
    cdef vector[uint32_t] cpp_columns_vector
    cdef vector[double] cpp_values_vector

    if type(tensor) == pd.DataFrame:
        idx_column_names = list(filter(lambda c: tensor[c].dtype==np.int64 or tensor[c].dtype==np.int32, tensor.columns))
        val_column_names = list(filter(lambda c: tensor[c].dtype==np.float32 or tensor[c].dtype==np.float64, tensor.columns))

        if len(val_column_names) != 1:
            error_msg = "tensor has {} float columns but must have exactly 1 value column.".format(len(val_column_names))
            raise ValueError(error_msg)

        idx = [i for c in idx_column_names for i in np.array(tensor[c], dtype=np.int32)]
        val = np.array(tensor[val_column_names[0]],dtype=np.float64)

        if shape is not None:
            cpp_dims_vector = shape
        else:
            cpp_dims_vector = [tensor[c].max() + 1 for c in idx_column_names]

        cpp_columns_vector = idx
        cpp_values_vector = val

        return new TensorConfig(make_shared[vector[uint64_t]](cpp_dims_vector), make_shared[vector[uint32_t]](cpp_columns_vector), make_shared[vector[double]](cpp_values_vector), NoiseConfig(), is_scarse)
    else:
        error_msg = "Unsupported sparse tensor data type: {}".format(tensor)
        raise ValueError(error_msg)

cdef TensorConfig* prepare_auxdata(in_tensor, shape):
    if type(in_tensor) in DENSE_TENSOR_TYPES:
        return prepare_dense_tensor(in_tensor)
    elif type(in_tensor) in SPARSE_TENSOR_TYPES:
        return prepare_sparse_tensor(in_tensor, shape, True)
    else:
        error_msg = "Unsupported tensor type: {}".format(type(in_tensor))
        raise ValueError(error_msg)

cdef (shared_ptr[TensorConfig], shared_ptr[TensorConfig]) prepare_data(train, test, shape):
    # Check train data type
    if (type(train) not in DENSE_MATRIX_TYPES and
        type(train) not in SPARSE_MATRIX_TYPES and
        type(train) not in DENSE_TENSOR_TYPES and
        type(train) not in SPARSE_TENSOR_TYPES):
        error_msg = "Unsupported train data type: {}".format(type(train))
        raise ValueError(error_msg)

    # Check test data type
    if test is not None:
        if (type(test) not in DENSE_MATRIX_TYPES and
            type(test) not in SPARSE_MATRIX_TYPES and
            type(test) not in DENSE_TENSOR_TYPES and
            type(test) not in SPARSE_TENSOR_TYPES):
            error_msg = "Unsupported test data type: {}".format(type(test))
            raise ValueError(error_msg)

    # Check train and test data for mismatch
    if test is not None:
        if ((type(train) in DENSE_MATRIX_TYPES or type(train) in SPARSE_MATRIX_TYPES) and
            (type(test) not in DENSE_MATRIX_TYPES and type(test) not in SPARSE_MATRIX_TYPES)):
            error_msg = "Train and test data must be the same type: {} != {}".format(type(train), type(test))
            raise ValueError(error_msg)
        if (type(train) in DENSE_MATRIX_TYPES or type(train) in SPARSE_MATRIX_TYPES) and train.shape != test.shape:
            raise ValueError("Train and test data must be the same shape: {} != {}".format(train.shape, test.shape))
        if type(train) in SPARSE_TENSOR_TYPES and train.ndim != test.ndim:
            raise ValueError("Train and test data must have the same number of dimensions: {} != {}".format(train.ndim, test.ndim))

    cdef TensorConfig* train_config
    cdef TensorConfig* test_config

    # Prepare train data
    if type(train) in DENSE_MATRIX_TYPES and len(train.shape) == 2:
        train_config = prepare_dense_matrix(train)
    elif type(train) in SPARSE_MATRIX_TYPES:
        train_config = prepare_sparse_matrix(train, True)
    elif type(train) in DENSE_TENSOR_TYPES and len(train.shape) > 2:
        train_config = prepare_dense_tensor(train)
    elif type(train) in SPARSE_TENSOR_TYPES:
        train_config = prepare_sparse_tensor(train, shape, True)
    else:
        error_msg = "Unsupported train data type: {}".format(type(train))
        raise ValueError(error_msg)

    # Prepare test data
    if test is not None:
        if type(test) in DENSE_MATRIX_TYPES and len(test.shape) == 2:
            test_config = prepare_dense_matrix(test)
        elif type(test) in SPARSE_MATRIX_TYPES:
            test_config = prepare_sparse_matrix(test, True)
        elif type(test) in DENSE_TENSOR_TYPES and len(test.shape) > 2:
            test_config = prepare_dense_tensor(test)
        elif type(test) in SPARSE_TENSOR_TYPES:
            test_config = prepare_sparse_tensor(test, shape, True)
        else:
            error_msg = "Unsupported test data type: {}".format(type(test))
            raise ValueError(error_msg)

    if test is not None:
        # Adjust train and test dims. Makes sense only for sparse tensors
        # It does not have any effect in other case
        if shape is None:
            for i in range(train_config.getDimsPtr().get().size()):
                max_dim = max(train_config.getDimsPtr().get().at(i), test_config.getDimsPtr().get().at(i))
                train_config.getDimsPtr().get()[0][i] = max_dim
                test_config.getDimsPtr().get()[0][i] = max_dim
        return shared_ptr[TensorConfig](train_config), shared_ptr[TensorConfig](test_config)
    else:
        return shared_ptr[TensorConfig](train_config), shared_ptr[TensorConfig]()


# Create and initialize smurff-cpp Config instance
def createConfig(
           Y              = None,
           Ytest          = None,
           data_shape     = None,
           priors         = [],
           side_info      = [],
           aux_data       = [],
           lambda_beta    = 10.0,
           num_latent     = 10,
           precision      = None,
           sn_init        = None,
           sn_max         = None,
           burnin         = 50,
           nsamples       = 400,
           tol            = 1e-6,
           direct         = True,
           seed           = None,
           threshold      = None,
           verbose        = True,
           init_model     = None,
           save_prefix    = None,
           save_extension = None,
           save_freq      = None,
           csv_status     = None):

    cdef Config config
    cdef NoiseConfig nc
    cdef shared_ptr[PVec] pos

    if precision is not None:
        nc.setNoiseType(fixed)
        nc.setPrecision(precision)

    if sn_init is not None and sn_max is not None:
        nc.setNoiseType(adaptive)
        nc.setSnInit(sn_init)
        nc.setSnMax(sn_max)

    if threshold is not None:
        nc.setNoiseType(probit)
        nc.setThreshold(threshold)
        config.setThreshold(threshold)
        config.setClassify(True)

    train, test = prepare_data(Y, Ytest, data_shape)
    train.get().setNoiseConfig(nc)

    config.setTrain(train)
    if Ytest is not None:
        test.get().setNoiseConfig(nc)
        config.setTest(test)

    if len(side_info) == 0:
        side_info = [None] * len(priors)

    if len(aux_data) == 0:
        aux_data = [None] * len(priors)

    for i in range(len(priors)):
        prior_type_str = priors[i]
        config.getPriorTypes().push_back(stringToPriorType(prior_type_str))

        prior_side_info = side_info[i]
        if prior_side_info is None:
            config.getSideInfo().push_back(shared_ptr[MatrixConfig]())
        else:
            config.getSideInfo().push_back(shared_ptr[MatrixConfig](prepare_sideinfo(prior_side_info)))

        prior_aux_data = aux_data[i]
        if prior_aux_data is not None:
            pos = make_shared[PVec](len(priors))
            for ad in prior_aux_data:
                if ad is not None:
                    if i == 0:
                        (&pos.get()[0].at(1))[0] += 1
                    elif i == 1:
                        (&pos.get()[0].at(0))[0] += 1
                    else:
                        (&pos.get()[0].at(i))[0] += 1
                    config.getAuxData().push_back(shared_ptr[TensorConfig](prepare_auxdata(ad, data_shape)))
                    config.getAuxData().back().get().setPos(pos.get()[0])

    config.setLambdaBeta(lambda_beta)
    config.setNumLatent(num_latent)
    config.setBurnin(burnin)
    config.setNSamples(nsamples)
    config.setTol(tol)
    config.setDirect(direct)

    if seed:
        config.setRandomSeedSet(True)
        config.setRandomSeed(seed)

    config.setVerbose(verbose)
    if quiet:          config.setVerbose(False)
    if init_model:     config.setModelInitType(stringToModelInitType(init_model))
    if save_prefix:    config.setSavePrefix(save_prefix)
    if save_extension: config.setSaveExtension(save_extension)
    if save_freq:      config.setSaveFreq(save_freq)
    if csv_status:     config.setCsvStatus(csv_status)

    return config


class Session:
    def __init__(self, session_ptr):
        self.ptr = session_ptr
        self.ptr.get().init()

    @classmethod
    def fromRootFile(cls, root_path):
        return cls(SessionFactory.create_py_session_from_root_path(root_path))

    @classmethod
    def fromConfig(cls, config):
        return cls(SessionFactory.create_py_session_from_config(config))

    def step(self):
        return self.ptr.get().step()

    def run(self):
        while self.step():
            pass

    def getResult(self):
        """ Create Python list of ResultItem from C++ vector of ResultItem """
        cpp_result_items_ptr = self.ptr.get().getResult()
        py_result_items = []

        if cpp_result_items_ptr:
            for i in range(cpp_result_items_ptr.get().size()):
                cpp_result_item_ptr = &(cpp_result_items_ptr.get().at(i))
                py_result_item_coords = []
                for coord_index in range(cpp_result_item_ptr.coords.size()):
                    coord = cpp_result_item_ptr.coords[coord_index]
                    py_result_item_coords.append(coord)
                py_result_item = ResultItem( tuple(py_result_item_coords)
                                           , cpp_result_item_ptr.val
                                           , cpp_result_item_ptr.pred_1sample
                                           , cpp_result_item_ptr.pred_avg
                                           , cpp_result_item_ptr.var
                                           , cpp_result_item_ptr.stds
                                           )
                py_result_items.append(py_result_item)

        return Result(py_result_items, self.ptr.get().getRmseAvg())
