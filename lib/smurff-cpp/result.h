#ifndef RESULT_H
#define RESULT_H

#include <memory>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <Utils/utils.h>

namespace smurff {

struct Model;
class Data;
 
template<typename Item, typename Compare>
double calc_auc(
                    const std::vector<Item> &predictions,
                    double threshold, 
                    const Compare &compare
               )
{
    auto sorted_predictions = predictions;
    std::sort(sorted_predictions.begin(), sorted_predictions.end(), compare);

    int num_positive = 0;
    int num_negative = 0;
    double auc = .0;
    for(auto &t : sorted_predictions) {
        int is_positive = t.val > threshold;
        int is_negative = !is_positive;
        num_positive += is_positive;
        num_negative += is_negative;
        auc += is_positive * num_negative;
    }

    auc /= num_positive;
    auc /= num_negative;
    return auc;
}

template<typename Item>
double calc_auc(const std::vector<Item> &predictions, double threshold)
{
    return calc_auc(predictions, threshold, [](const Item &a, const Item &b) { return a.pred < b.pred;});
}


struct Result {
    //-- test set
    struct Item {
        int row, col;
        double val, pred_1sample, pred_avg, var, stds;
    };
    std::vector<Item> predictions;
    int nrows, ncols;
    void set(Eigen::SparseMatrix<double> Y);


    //-- prediction metrics
    void update(const Model &, const Data &, bool burnin);
    double rmse_avg = NAN;
    double rmse_1sample = NAN;
    double auc_avg = NAN;
    double auc_1sample = NAN;
    int sample_iter = 0;
    int burnin_iter = 0;

    double rmse_using_globalmean(double);
    double rmse_using_modemean(const Data &data, int mode);

    // general
    void save(std::string fname_prefix);
    void init();
    std::ostream &info(std::ostream &os, std::string indent, const Data &data);

    //-- for binary classification
    int total_pos = -1;
    bool classify = false;
    double threshold;
    void setThreshold(double t) { threshold = t; classify = true; }
};

}; // end namespace smurff

#endif
