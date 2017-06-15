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

#include "data.h"
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


double Result::rmse_using_globalmean(double mean) {
    double se = 0.;
    for(auto t : predictions) se += square(t.val - mean);
    return sqrt( se / predictions.size() );
}

double Result::rmse_using_modemean(const Data &data, int mode) {
     const unsigned N = predictions.size();
     double se = 0.;
     for(auto t : predictions) {
        int n = mode == 0 ? t.col : t.row;
        double pred = data.mean(mode, n);
        se += square(t.val - pred);
     }
     return sqrt( se / N );
}

//--- output model to files
void Result::save(std::string prefix) {
    if (predictions.empty()) return;
    std::string fname_pred = prefix + "-predictions.csv";
    std::ofstream predfile;
    predfile.open(fname_pred);
    predfile << "row,col,y,pred_1samp,pred_avg,var,std\n";
    for ( auto &t : predictions) {
        predfile
                << to_string( t.row  )
         << "," << to_string( t.col  )
         << "," << to_string( t.val  )
         << "," << to_string( t.pred )
         << "," << to_string( t.pred_avg )
         << "," << to_string( t.var )
         << "," << to_string( t.stds )
         << "\n";
    }
    predfile.close();

}

///--- update RMSE and AUC

void Result::update(const Model &model, const Data &data,  bool burnin)
{
    if (predictions.size() == 0) return;
    const unsigned N = predictions.size();

    if (burnin) {
        double se = 0.0;
#pragma omp parallel for schedule(guided) reduction(+:se)
        for(unsigned k=0; k<predictions.size(); ++k) {
            auto &t = predictions[k];
            t.pred = model.dot({t.col, t.row}) + data.offset_to_mean({t.col, t.row});
            se += square(t.val - t.pred);
        }
        burnin_iter++;
        rmse = sqrt( se / N );
    } else {
        double se = 0.0, se_avg = 0.0;
#pragma omp parallel for schedule(guided) reduction(+:se, se_avg)
        for(unsigned k=0; k<predictions.size(); ++k) {
            auto &t = predictions[k];
            const double pred = model.dot({t.col, t.row}) + data.offset_to_mean({t.col, t.row});
            se += square(t.val - pred);
            double delta = pred - t.pred_avg;
            double pred_avg = (t.pred_avg + delta / (sample_iter + 1));
            t.var += delta * (pred - pred_avg);
            const double inorm = 1.0 / sample_iter;
            t.stds = sqrt(t.var * inorm);
            t.pred_avg = pred_avg;
            t.pred = pred;
            se_avg += square(t.val - pred_avg);
        }
        sample_iter++;
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

std::ostream &Result::info(std::ostream &os, std::string indent, const Data &data)
{
    if (predictions.size()) {
        double test_fill_rate = 100. * predictions.size() / nrows / ncols;
        os << indent << "Test data: " << predictions.size() << " [" << nrows << " x " << ncols << "] (" << test_fill_rate << "%)\n";
        os << indent << "RMSE using globalmean: " << rmse_using_globalmean(data.mean()) << endl;
        os << indent << "RMSE using colmean: " << rmse_using_modemean(data,0) << endl;
        os << indent << "RMSE using rowmean: " << rmse_using_modemean(data,1) << endl;
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
