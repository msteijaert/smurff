from libcpp cimport bool
from libcpp.string cimport string
from libcpp.vector cimport vector

cdef extern from "model.h" namespace "smurff":
    cdef cppclass Result:
        cppclass Item:
            int row, col
            double val, pred, var, stds

        vector[Item] predictions
        int nrows, ncols

        double rmse_avg
        double rmse
        double auc

cdef extern from "session.h" namespace "smurff":
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
        MatrixConfig config_train, config_test
        string fname_train
        string fname_test
        double test_split         

        #-- features
        vector[MatrixConfig] config_row_features
        vector[string] fname_row_features
        vector[MatrixConfig] config_col_features
        vector[string] fname_col_features

        # -- priors
        string row_prior 
        string col_prior 

        #-- output
        string output_prefix

        #-- general
        bool verbose              
        int output_freq           
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



    cdef cppclass PythonSession:
        PythonSession()
        void setFromConfig(Config)
        void step()
        void init()
        Result pred

