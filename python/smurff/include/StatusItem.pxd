from libcpp.vector cimport vector
from libcpp.string cimport string

cdef extern from "<SmurffCpp/StatusItem.h>" namespace "smurff":
    cdef cppclass StatusItem:
        string phase;
        int iter;
        int phase_iter;

        vector[double] model_norms;

        double rmse_avg;
        double rmse_1sample;
        double train_rmse;

        double auc_1sample;
        double auc_avg;

        double elapsed_iter;
        double nnz_per_sec;
        double samples_per_sec;