from libcpp cimport bool
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.memory cimport shared_ptr

from MatrixConfig cimport MatrixConfig
from TensorConfig cimport TensorConfig

cdef extern from "<SmurffCpp/Configs/Config.h>" namespace "smurff":
    cdef cppclass PriorTypes:
        pass

    cdef cppclass ModelInitTypes:
        pass

    PriorTypes stringToPriorType(string name)
    ModelInitTypes stringToModelInitType(string name)

    cdef cppclass Config:
        #-- train and test
        shared_ptr[TensorConfig] m_train
        shared_ptr[TensorConfig] m_test

        #-- features
        vector[shared_ptr[MatrixConfig]] m_row_features
        vector[shared_ptr[MatrixConfig]] m_col_features

        #-- priors
        PriorTypes row_prior_type
        PriorTypes col_prior_type

        #-- restore
        string restore_prefix
        string restore_suffix

        #-- init model
        ModelInitTypes model_init_type

        #-- save
        string getSavePrefix()
        void setSavePrefix(string value)

        string save_suffix
        int save_freq

        #-- general
        bool random_seed_set
        int random_seed
        int verbose
        string csv_status
        int burnin
        int nsamples
        int num_latent

        #-- for macau priors
        double lambda_beta
        double tol
        bool direct

        #-- binary classification
        bool classify
        double threshold;
