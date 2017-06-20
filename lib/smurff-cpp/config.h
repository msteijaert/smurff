#pragma once

#include <cassert>
#include <string>
#include <vector>


namespace smurff {

struct MatrixConfig {
    MatrixConfig()
        : dense(true), binary(false), rows(0), cols(0), values(0), nnz(0), nrow(0), ncol(0) {}
    MatrixConfig(int nrow, int ncol, double *values)
        : dense(true), binary(false), rows(0), cols(0), values(values), nnz(nrow*ncol), nrow(nrow), ncol(ncol) {}
    MatrixConfig(int nrow, int ncol, int nnz, int *rows, int *cols, double *values)
        : dense(false), binary(false), rows(rows), cols(cols), values(values), nnz(nnz), nrow(nrow), ncol(ncol) {}
    MatrixConfig(int nrow, int ncol, int nnz, int *rows, int *cols)
        : dense(false), binary(true), rows(rows), cols(cols), values(0), nnz(nnz), nrow(nrow), ncol(ncol) {}
    MatrixConfig(int nrow, int ncol, bool dense, bool binary)
        : dense(dense), binary(binary), rows(0), cols(0), values(0), nnz(0), nrow(nrow), ncol(ncol) {}
    MatrixConfig(int nrow, int ncol, int nnz, bool binary = false)
        : dense(nnz == nrow*ncol), binary(binary), rows(0), cols(0), values(0), nnz(nnz), nrow(nrow), ncol(ncol) {}

    bool dense;
    bool binary;

    int* rows;
    int* cols;
    double* values;
    int nnz;
    int nrow;
    int ncol;

    void alloc() {
        assert(!cols);
        assert(!rows);
        assert(!values);
        if (!dense) {
            rows = new int[nnz];
            cols = new int[nnz];
        }
        values = new double[nnz];
    }

    std::ostream &info(std::ostream &) const;
};

struct Config {
    
    //-- train and test
    MatrixConfig train, test;
    double test_split         = .0;
    std::string center_mode   = "global";

    //-- features
    std::vector<MatrixConfig> row_features;
    std::vector<MatrixConfig> col_features;

    // -- priors
    std::string row_prior = "default";
    std::string col_prior = "default";
    
    //-- restore
    std::string restore_prefix = "";
    std::string restore_suffix = ".csv";

    //-- init model
    std::string init_model = "zero";

    //-- save
    std::string save_prefix = "save";
    std::string save_suffix = ".csv";
    int save_freq           = 0; // never

    //-- general
    int verbose               = 1;
    std::string csv_status    = "";
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
    void save(std::string) const;
    void restore(std::string);
    static std::string version();
};

} // end namespace smurff

