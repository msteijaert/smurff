#pragma once

#include <string>
#include <vector>


namespace Macau {

struct MatrixConfig {
    MatrixConfig()
        : dense(true), rows(0), cols(0), values(0), nnz(0), nrow(0), ncol(0) {}
    MatrixConfig(int nrow, int ncol, double *values)
        : dense(true), rows(0), cols(0), values(values), nnz(nrow*ncol), nrow(nrow), ncol(ncol) {}

    MatrixConfig(int nrow, int ncol, int nnz, int *rows, int *cols, double *values)
        : dense(false), rows(rows), cols(cols), values(values), nnz(nnz), nrow(nrow), ncol(ncol) {}

    bool dense;
    int* rows;
    int* cols;
    double* values;
    int nnz;
    int nrow;
    int ncol;
};

struct Config {
    
    //-- train and test
    MatrixConfig config_train, config_test;
    std::string fname_train;
    std::string fname_test;
    double test_split         = .0;

    //-- features
    std::vector<MatrixConfig> config_row_features;
    std::vector<std::string> fname_row_features;
    std::vector<MatrixConfig> config_col_features;
    std::vector<std::string> fname_col_features;

    // -- priors
    std::string row_prior = "default";
    std::string col_prior = "default";

    //-- intialize U
    std::string fname_row_model;
    std::string fname_col_model;

    //-- output
    std::string output_prefix = "save";

    //-- general
    bool verbose              = false;
    int output_freq           = 0; // never
    int burnin                = 200;
    int nsamples              = 800;
    int num_latent            = 96;

    //-- for macau priors
    double lambda_beta        = 10.0;
    double tol                = 1e-6;
    bool direct               = false; 

    //-- noise model
    std::string noise_model   = "fixed";
    double precision          = 5.0;
    double sn_init            = 1.0;
    double sn_max             = 10.0;

    //-- binary classification
    bool classify             = false;
    double threshold;

    bool validate(bool) const;
};

} // end namespace Macau

