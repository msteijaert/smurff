#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <memory>
#include <cmath>

#include "mvnormal.h"
#include "noisemodels.h"
#include "model.h"

using namespace Eigen;

////  AdaptiveGaussianNoise  ////
void AdaptiveGaussianNoise::init() {
    double se = 0.0;
    const Eigen::SparseMatrix<double> &train = model.Y();
    const double mean_rating = model.mean_rating();

#pragma omp parallel for schedule(dynamic, 4) reduction(+:se)
    for (int k = 0; k < train.outerSize(); ++k) {
        for (SparseMatrix<double>::InnerIterator it(train,k); it; ++it) {
            se += square(it.value() - mean_rating);
        }
    }

    var_total = se / train.nonZeros();
    if (var_total <= 0.0 || std::isnan(var_total)) {
        // if var cannot be computed using 1.0
        var_total = 1.0;
    }
    // Var(noise) = Var(total) / (SN + 1)
    alpha     = (sn_init + 1.0) / var_total;
    alpha_max = (sn_max + 1.0) / var_total;
}

void AdaptiveGaussianNoise::update()
{
    double sumsq = 0.0;
    const Eigen::SparseMatrix<double> &train = model.Y();
    const double mean_rating = model.mean_rating();

#pragma omp parallel for schedule(dynamic, 4) reduction(+:sumsq)
    for (int j = 0; j < train.outerSize(); j++) {
        auto Vj = model.col(1, j);
        for (SparseMatrix<double>::InnerIterator it(train, j); it; ++it) {
            double Yhat = Vj.dot( model.col(0, it.row()) ) + mean_rating;
            sumsq += square(Yhat - it.value());
        }
    }
    // (a0, b0) correspond to a prior of 1 sample of noise with full variance
    double a0 = 0.5;
    double b0 = 0.5 * var_total;
    double aN = a0 + train.nonZeros() / 2.0;
    double bN = b0 + sumsq / 2.0;
    alpha = rgamma(aN, 1.0 / bN);
    if (alpha > alpha_max) {
        alpha = alpha_max;
    }
}

 // Evaluation metrics
void FixedGaussianNoise::evalModel(bool burnin) {
   model.update_rmse(burnin);
   rmse_test = model.rmse_avg;
   rmse_test_onesample = model.rmse;
}

void AdaptiveGaussianNoise::evalModel(bool burnin) {
   model.update_rmse(burnin);
   rmse_test = model.rmse_avg;
   rmse_test_onesample = model.rmse;
}

void ProbitNoise::evalModel(bool burnin) {
    model.update_auc(burnin);
    auc_test_onesample = model.auc;
    auc_test = model.auc_avg;
}

std::pair<double, double> ProbitNoise::sample(int n, int m) {
    double y = model.Y().coeffRef(n,m);
    const VectorXd &u = model.col(0, n);
    const VectorXd &v = model.col(1, n);
    double z = (2 * y - 1) * fabs(v.dot(u) + bmrandn_single());
    return std::make_pair(1, z);
}
