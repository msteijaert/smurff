##
#
# Cython file that bridges between the smurff python code and the C++ code
# 
# This file will be compiled via CMake
#


from libc.stdint cimport *
from libcpp.memory cimport shared_ptr, make_shared
from libcpp.vector cimport vector
from libcpp.string cimport string

from Config cimport *
from ISession cimport ISession
from NoiseConfig cimport *
from MatrixConfig cimport MatrixConfig
from TensorConfig cimport TensorConfig
from SideInfoConfig cimport *
from SessionFactory cimport SessionFactory
from PVec cimport PVec

cimport numpy as np
import  numpy as np
import  scipy as sp
import pandas as pd
import scipy.sparse
import numbers

from .prepare import make_train_test, make_train_test_df
from .result import Result, ResultItem

DENSE_MATRIX_TYPES  = (np.ndarray, np.matrix, )
SPARSE_MATRIX_TYPES = (sp.sparse.coo.coo_matrix, sp.sparse.csr.csr_matrix, sp.sparse.csc.csc_matrix, )

MATRIX_TYPES = DENSE_MATRIX_TYPES + SPARSE_MATRIX_TYPES

DENSE_TENSOR_TYPES  = (np.ndarray, )
SPARSE_TENSOR_TYPES = (pd.DataFrame, )

TENSOR_TYPES = DENSE_TENSOR_TYPES + SPARSE_TENSOR_TYPES

ALL_TYPES = MATRIX_TYPES + TENSOR_TYPES

def remove_nan(Y):
    if not np.any(np.isnan(Y.data)):
        return Y
    idx = np.where(np.isnan(Y.data) == False)[0]
    return sp.sparse.coo_matrix( (Y.data[idx], (Y.row[idx], Y.col[idx])), shape = Y.shape )

cdef MatrixConfig* prepare_sparse_matrix(X, NoiseConfig noise_config, is_scarse) except +:
    if not isinstance(X, SPARSE_MATRIX_TYPES):
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

    cdef MatrixConfig* matrix_config_ptr = new MatrixConfig(<uint64_t>(X.shape[0]), <uint64_t>(X.shape[1]), rows_vector_shared_ptr, cols_vector_shared_ptr, vals_vector_shared_ptr, noise_config, is_scarse)
    return matrix_config_ptr

cdef MatrixConfig* prepare_dense_matrix(X, NoiseConfig noise_config) except +:
    cdef np.ndarray[np.double_t] vals = np.squeeze(np.asarray(X.flatten(order='F')))
    cdef double* vals_begin = &vals[0]
    cdef double* vals_end = vals_begin + vals.shape[0]
    cdef vector[double]* vals_vector_ptr = new vector[double]()
    vals_vector_ptr.assign(vals_begin, vals_end)
    cdef shared_ptr[vector[double]] vals_vector_shared_ptr = shared_ptr[vector[double]](vals_vector_ptr)
    cdef MatrixConfig* matrix_config_ptr = new MatrixConfig(<uint64_t>(X.shape[0]), <uint64_t>(X.shape[1]), vals_vector_shared_ptr, noise_config)
    return matrix_config_ptr

cdef shared_ptr[SideInfoConfig] prepare_sideinfo(side_info, NoiseConfig noise_config, tol, direct) except +:
    if isinstance(side_info[0], SPARSE_MATRIX_TYPES):
        side_info_config_matrix = prepare_sparse_matrix(side_info[0], noise_config, False)
    elif isinstance(side_info[0], DENSE_MATRIX_TYPES) and len(side_info[0].shape) == 2:
        side_info_config_matrix = prepare_dense_matrix(side_info[0], noise_config)
    else:
        error_msg = "Unsupported side info matrix type: {}".format(side_info[0])
        raise ValueError(error_msg)

    cdef shared_ptr[SideInfoConfig] side_info_config_ptr = make_shared[SideInfoConfig]()
    side_info_config_ptr.get().setSideInfo(shared_ptr[MatrixConfig](side_info_config_matrix))
    side_info_config_ptr.get().setTol(tol)
    side_info_config_ptr.get().setDirect(direct)
    return side_info_config_ptr

cdef TensorConfig* prepare_dense_tensor(tensor, NoiseConfig noise_config) except +:
    if not isinstance(tensor, DENSE_TENSOR_TYPES):
        error_msg = "Unsupported dense tensor data type: {}".format(tensor)
        raise ValueError(error_msg)
    raise NotImplementedError()

cdef TensorConfig* prepare_sparse_tensor(tensor, shape, NoiseConfig noise_config, is_scarse) except +:
    if not isinstance(tensor, SPARSE_TENSOR_TYPES):
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

cdef TensorConfig* prepare_auxdata(in_tensor, shape, NoiseConfig noise_config) except +:
    if isinstance(in_tensor, DENSE_TENSOR_TYPES):
        return prepare_dense_tensor(in_tensor, noise_config)
    elif isinstance(in_tensor, SPARSE_TENSOR_TYPES):
        return prepare_sparse_tensor(in_tensor, shape, noise_config, True)
    else:
        error_msg = "Unsupported tensor type: {}".format(type(in_tensor))
        raise ValueError(error_msg)

cdef (shared_ptr[TensorConfig], shared_ptr[TensorConfig]) prepare_data(train, test, shape, NoiseConfig noise_config) except +:
    # Check train data type
    if (not isinstance(train, ALL_TYPES)):
        error_msg = "Unsupported train data type: {}".format(type(train))
        raise ValueError(error_msg)

    # Check test data type
    if test is not None:
        if (not isinstance(test, ALL_TYPES)):
            error_msg = "Unsupported test data type: {}".format(type(test))
            raise ValueError(error_msg)

    # Check train and test data for mismatch
    if test is not None:
        if isinstance(train, MATRIX_TYPES) and (not isinstance(test, MATRIX_TYPES)):
            error_msg = "Train and test data must be the same type: {} != {}".format(type(train), type(test))
            raise ValueError(error_msg)
        if isinstance(train, MATRIX_TYPES) and train.shape != test.shape:
            raise ValueError("Train and test data must be the same shape: {} != {}".format(train.shape, test.shape))
        if isinstance(train, SPARSE_TENSOR_TYPES) and train.ndim != test.ndim:
            raise ValueError("Train and test data must have the same number of dimensions: {} != {}".format(train.ndim, test.ndim))

    cdef TensorConfig* train_config
    cdef TensorConfig* test_config

    # Prepare train data
    if isinstance(train, DENSE_MATRIX_TYPES) and len(train.shape) == 2:
        train_config = prepare_dense_matrix(train, noise_config)
    elif isinstance(train, SPARSE_MATRIX_TYPES):
        train_config = prepare_sparse_matrix(train, noise_config, True)
    elif isinstance(train, DENSE_TENSOR_TYPES) and len(train.shape) > 2:
        train_config = prepare_dense_tensor(train, noise_config)
    elif isinstance(train, SPARSE_TENSOR_TYPES):
        train_config = prepare_sparse_tensor(train, shape, noise_config, True)
    else:
        error_msg = "Unsupported train data type or shape: {}".format(type(train))
        raise ValueError(error_msg)
    train_config.setNoiseConfig(noise_config)

    # Prepare test data
    if test is not None:
        if isinstance(test, DENSE_MATRIX_TYPES) and len(test.shape) == 2:
            test_config = prepare_dense_matrix(test, noise_config)
        elif isinstance(test, SPARSE_MATRIX_TYPES):
            test_config = prepare_sparse_matrix(test, noise_config, True)
        elif isinstance(test, DENSE_TENSOR_TYPES) and len(test.shape) > 2:
            test_config = prepare_dense_tensor(test, noise_config)
        elif isinstance(test, SPARSE_TENSOR_TYPES):
            test_config = prepare_sparse_tensor(test, shape, noise_config, True)
        else:
            error_msg = "Unsupported test data type: {} or shape {}".format(type(test), test.shape)
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

class PyNoiseConfig:
    def __init__(self, noise_type = "fixed", precision = 5.0, sn_init = 1.0, sn_max = 10.0, probit_threshold = 0.5): 
        self.noise_type = noise_type
        self.precision = precision
        self.sn_init = sn_init
        self.probit_threshold = probit_threshold

cdef NoiseConfig prepare_noise_config(py_noise_config):
    cdef NoiseConfig n
    n.setNoiseType(py_noise_config.noise_type.encode('UTF-8'))
    return n


cdef class PySession:
    cdef shared_ptr[ISession] ptr;
    cdef Config config
    cdef NoiseConfig noise_config

    def __init__(self,
        priors           = [ "normal", "normal" ],
        num_latent       = NUM_LATENT_DEFAULT_VALUE,
        burnin           = BURNIN_DEFAULT_VALUE,
        nsamples         = NSAMPLES_DEFAULT_VALUE,
        tol              = TOL_DEFAULT_VALUE,
        direct           = True,
        seed             = RANDOM_SEED_DEFAULT_VALUE,
        verbose          = VERBOSE_DEFAULT_VALUE,
        save_prefix      = SAVE_PREFIX_DEFAULT_VALUE,
        save_extension   = SAVE_EXTENSION_DEFAULT_VALUE,
        save_freq        = SAVE_FREQ_DEFAULT_VALUE,
        csv_status       = STATUS_DEFAULT_VALUE):

        for p in priors:
            self.config.addPriorType(p.encode('UTF-8'))
        self.config.setNumLatent(num_latent)
        self.config.setBurnin(burnin)
        self.config.setNSamples(nsamples)
        self.config.setVerbose(verbose)

        if seed:           self.config.setRandomSeed(seed)
        if save_prefix:    self.config.setSavePrefix(save_prefix)
        if save_extension: self.config.setSaveExtension(save_extension)
        if save_freq:      self.config.setSaveFreq(save_freq)
        if csv_status:     self.config.setCsvStatus(csv_status)

    def addTrainAndTest(self, Y, Ytest = None, noise = PyNoiseConfig()):
        self.noise_config = prepare_noise_config(noise)
        train, test = prepare_data(Y, Ytest, None, self.noise_config)
        self.config.setTrain(train)

        if Ytest is not None:
            self.config.setTest(test)

    def addSideInfo(self, mode, Y, noise = PyNoiseConfig(), tol = 1e-6, direct = False):
        self.noise_config = prepare_noise_config(noise)
        self.config.addSideInfoConfig(mode, prepare_sideinfo(Y, self.noise_config, tol, direct))

    def init(self):
        self.ptr = SessionFactory.create_py_session_from_config(self.config)
        return self.ptr.get().init()

    def step(self):
        return self.ptr.get().step()

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
