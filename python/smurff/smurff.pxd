from libcpp cimport bool
from libcpp.string cimport string
from libcpp.vector cimport vector

cdef extern from "result.h" namespace "smurff":
    cdef cppclass Result:
        cppclass Item:
            int row, col
            double val, pred, var, stds

        vector[Item] predictions
        int nrows, ncols

        double rmse_avg
        double rmse
        double auc

cdef extern from "config.h" namespace "smurff":
    cdef cppclass MatrixConfig:
        MatrixConfig()
        MatrixConfig(int nrows, int ncols, double *values)
        MatrixConfig(int nrows, int ncols, int N, int *rows, int *cols, double *values)
                
        int* rows
        int* cols
        int N
        double* values
        int nrows
        int ncols
        bool dense

    cdef cppclass Config:
        #-- train and test
        MatrixConfig train, test
        double test_split         

        #-- features
        vector[MatrixConfig] row_features
        vector[MatrixConfig] col_features

        # -- priors
        string row_prior 
        string col_prior 

        #-- output
        string save_prefix

        #-- general
        int verbose              
        int save_freq           
        int burnin                
        int nsamples              
        int num_latent            
        double lambda_beta        
        double tol                

        #-- noise model
        string fixed_precision, adaptive_precision
        double precision          
        double sn_init            
        double sn_max             

        #-- binary classification
        bool classify             
        double threshold

        @staticmethod
        string version()


cdef extern from "session.h" namespace "smurff":
    cdef cppclass PythonSession:
        PythonSession()
        void setFromConfig(Config)
        void step()
        void init()
        Result pred

