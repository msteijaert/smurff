#pragma once

#include <string>
#include <vector>

namespace smurff {

struct StatusItem
{
    std::string phase;
    int iter;
    int phase_iter;

    std::vector<float> model_norms;

    float rmse_avg;
    float rmse_1sample;
    float train_rmse;

    float auc_1sample;
    float auc_avg;

    float elapsed_iter;
    float elapsed_total;

    float nnz_per_sec;
    float samples_per_sec;

    // to csv 
    static std::string getCsvHeader();
    std::string asCsvString() const;

    std::string asString() const;
};
}