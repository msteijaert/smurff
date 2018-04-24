#pragma once

#include <string>

namespace smurff {

struct StatusItem
{
    std::string phase;
    int iter;
    int phase_iter;

    std::vector<double> model_norms;

    double rmse_avg;
    double rmse_1sample;
    double train_rmse;

    double auc_1sample;
    double auc_avg;

    double elapsed_iter;
    double nnz_per_sec;
    double samples_per_sec;

    // to csv 
    static std::string getCsvHeader();
    std::string asCsvString() const;
};
}