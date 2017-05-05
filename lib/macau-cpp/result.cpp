#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <iostream>
#include <cassert>
#include <fstream>
#include <string>
#include <algorithm>
#include <random>
#include <chrono>
#include <memory>
#include <cmath>

#include "model.h"
#include "utils.h"
#include "result.h"

using namespace std; 
using namespace Eigen;

namespace Macau {

void Result::set(SparseMatrixD Y) {
    for (int k = 0; k < Y.outerSize(); ++k) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(Y,k); it; ++it) {
            predictions.push_back({(int)it.row(), (int)it.col(), it.value()});
        }
    }
    nrows = Y.rows();
    ncols = Y.cols();
    init();
}

void Result::init() {
    total_pos = 0;
    if (classify) {
        for(auto &t : predictions) {
            int is_positive = t.val > threshold;
            total_pos += is_positive;
        }
    }
}

//--- output model to files
void Result::save(std::string prefix) {
    if (predictions.empty()) return;
    std::string fname_pred = prefix + "-predictions.csv";
    std::ofstream predfile;
    predfile.open(fname_pred);
    predfile << "row,col,y,y_pred,y_pred_std\n";
    for ( auto &t : predictions) {
        predfile
                << to_string( t.row  )
         << "," << to_string( t.col  )
         << "," << to_string( t.val  )
         << "," << to_string( t.pred )
         << "," << to_string( t.stds )
         << "\n";
    }
    predfile.close();

}

///--- update RMSE and AUC

void Result::update(const Model &model, bool burnin)
{
    if (predictions.size() == 0) return;
    const unsigned N = predictions.size();

    if (burnin) {
        double se = 0.0;
#pragma omp parallel for schedule(guided) reduction(+:se)
        for(unsigned k=0; k<predictions.size(); ++k) {
            auto &t = predictions[k];
            t.pred = model.predict(t.row, t.col);
            se += square(t.val - t.pred);
            burnin_iter++;
        }
        rmse = sqrt( se / N );
    } else {
        double se = 0.0, se_avg = 0.0;
#pragma omp parallel for schedule(guided) reduction(+:se, se_avg)
        for(unsigned k=0; k<predictions.size(); ++k) {
            auto &t = predictions[k];
            const double pred = model.predict(t.row, t.col);
            se += square(t.val - pred);
            double delta = pred - t.pred;
            double pred_avg = (t.pred + delta / (sample_iter + 1));
            t.var += delta * (pred - pred_avg);
            const double inorm = 1.0 / sample_iter;
            t.stds = sqrt(t.var * inorm);
            t.pred = pred_avg;
            se_avg += square(t.val - pred_avg);
            sample_iter++;
        }
        rmse = sqrt( se / N );
        rmse_avg = sqrt( se_avg / N );
    }

    update_auc();
}


void Result::update_auc()
{
    if (!classify) return;
    std::sort(predictions.begin(), predictions.end(),
            [this](const Item &a, const Item &b) { return a.pred < b.pred;});

    int num_positive = 0;
    int num_negative = 0;
    auc = .0;
    for(auto &t : predictions) {
        int is_positive = t.val > threshold;
        int is_negative = !is_positive; 
        num_positive += is_positive;
        num_negative += is_negative;
        auc += is_positive * num_negative;
    }

    auc /= num_positive;
    auc /= num_negative;
}

std::ostream &Result::info(std::ostream &os, std::string indent)
{
    if (predictions.size()) {
        double test_fill_rate = 100. * predictions.size() / nrows / ncols;
        os << indent << "Test data: " << predictions.size() << " [" << nrows << " x " << ncols << "] (" << test_fill_rate << "%)\n";
    } else {
        os << indent << "Test data: -\n";
    }
    if (classify) {
        double pos = 100. * (double)total_pos / (double)predictions.size();
        os << indent << "Binary classification threshold: " << threshold << "\n";
        os << indent << "  " << pos << "% positives in test data\n";
    }
    return os;
}

} // end namespace
