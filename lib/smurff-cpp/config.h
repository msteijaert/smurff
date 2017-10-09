#pragma once

#include <cassert>
#include <string>
#include <vector>

#include "MatrixConfig.h"

namespace smurff {

struct Config 
{
    //-- train and test
    MatrixConfig train, test;
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

    //-- binary classification
    bool classify             = false;
    double threshold;

    bool validate(bool = true) const;
    void save(std::string) const;
    void restore(std::string);
    static std::string version();
};

} // end namespace smurff

