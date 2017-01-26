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

#include <unsupported/Eigen/SparseExtra>
#include <Eigen/Sparse>

#if defined(_OPENMP)
#include <omp.h>
#endif

#include <signal.h>

#include "dsparse.h"
#include "utils.h"
#include "model.h"
#include "mvnormal.h"

using namespace std; 
using namespace Eigen;

void Factors::setRelationDataTest(int* rows, int* cols, double* values, int N, int nrows, int ncols) {
  assert(nrows == Yrows() && ncols == Ycols() && 
         "Size of train must be equal to size of test");
    
  Ytest.resize(nrows, ncols);
  sparseFromIJV(Ytest, rows, cols, values, N);
}
 
void Factors::setRelationDataTest(SparseDoubleMatrix &Y) {
   setRelationDataTest(Y.rows, Y.cols, Y.vals, Y.nnz, Y.nrow, Y.ncol);
}
 
void Factors::setRelationDataTest(SparseMatrixD Y) {
    Ytest = Y;
}

//--- output model to files

void Factors::savePredictions(std::string save_prefix, int iter, int burnin) {
    VectorXd yhat_sd_raw  = getStds(iter, burnin);
    MatrixXd testdata_raw = to_coo(Ytest);

    std::string fname_pred = save_prefix + "-predictions.csv";
    std::ofstream predfile;
    predfile.open(fname_pred);
    predfile << "row,col,y,y_pred,y_pred_std\n";
    for (int i = 0; i < predictions.size(); i++) {
        predfile << to_string( (int)testdata_raw(i,0) );
        predfile << "," << to_string( (int)testdata_raw(i,1) );
        predfile << "," << to_string( testdata_raw(i,2) );
        predfile << "," << to_string( predictions(i) );
        predfile << "," << to_string( yhat_sd_raw(i) );
        predfile << "\n";
    }
    predfile.close();
    printf("Saved predictions into '%s'.\n", fname_pred.c_str());

}

void Factors::saveGlobalParams(std::string save_prefix) {
  VectorXd means(1);
  means << mean_rating;
  writeToCSVfile(save_prefix + "-meanvalue.csv", means);
}

void Factors::saveModel(std::string save_prefix, int iter, int burnin) {
    int i = 0;
    for(auto &U : factors) {
        writeToCSVfile(save_prefix + "U" + std::to_string(++i) + "-latents.csv", U);
    }
    savePredictions(save_prefix, iter, burnin);
}

///--- update RMSE and AUC

void Factors::update_predictions(int iter, int burnin)
{
    if (Ytest.nonZeros() == 0) return;
    assert(last_iter <= iter);

    if (last_iter < 0) {
        const int N = Ytest.nonZeros();
        predictions     = VectorXd::Zero(N);
        predictions_var = VectorXd::Zero(N);
        stds            = VectorXd::Zero(N);
    }

    if (last_iter == iter) return;

    assert(last_iter + 1 == iter);

    double se = 0.0, se_avg = 0.0;
    const double inorm = 1.0 / (iter - burnin - 1);
#pragma omp parallel for schedule(dynamic,8) reduction(+:se, se_avg)
    for (int k = 0; k < Ytest.outerSize(); ++k) {
        int idx = Ytest.outerIndexPtr()[k];
        for (Eigen::SparseMatrix<double>::InnerIterator it(Ytest,k); it; ++it) {
            const double pred = col(0,it.col()).dot(col(1,it.row())) + mean_rating;
            se += square(it.value() - pred);

            // https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online_algorithm
            double pred_avg;
            if (iter <= burnin) {
                pred_avg = pred;
            } else {
                double delta = pred - predictions[idx];
                pred_avg = (predictions[idx] + delta / (iter - burnin + 1));
                predictions_var[idx] += delta * (pred - pred_avg);
            }
            se_avg += square(it.value() - pred_avg);
            predictions[idx] = pred_avg;
            stds[idx] = sqrt(predictions_var[idx] * inorm);
            idx++;
        }
    }

    const unsigned N = Ytest.nonZeros();
    rmse = sqrt( se / N );
    rmse_avg = sqrt( se_avg / N );
    last_iter = iter;
}

std::pair<double,double> Factors::getRMSE(int iter, int burnin)
{
    update_predictions(iter, burnin);
    return std::make_pair(rmse, rmse_avg);
}

const Eigen::VectorXd &Factors::getPredictions(int iter, int burnin)
{
    update_predictions(iter, burnin);
    return predictions;
}

const Eigen::VectorXd &Factors::getPredictionsVar(int iter, int burnin)
{
    update_predictions(iter, burnin);
    return predictions_var;
}

const Eigen::VectorXd &Factors::getStds(int iter, int burnin)
{
    update_predictions(iter, burnin);
    return stds;
}

double Factors::auc()
{
    if (Ytest.nonZeros() == 0) return NAN;

    double *test_vector = Ytest.valuePtr();

    Eigen::VectorXd stack_x(predictions.size());
    Eigen::VectorXd stack_y(predictions.size());
    double auc = 0.0;

    std::vector<unsigned int> permutation( predictions.size() );
    for(unsigned int i = 0; i < predictions.size(); i++) {
        permutation[i] = i;
    }
    std::sort(permutation.begin(), permutation.end(), [this](unsigned int a, unsigned int b) { return predictions[a] < predictions[b];});

    int NP = Ytest.sum();
    int NN = Ytest.nonZeros() - NP;
    //Build stack_x and stack_y
    stack_x[0] = test_vector[permutation[0]];
    stack_y[0] = 1-stack_x[0];
    for(int i=1; i < predictions.size(); i++) {
        stack_x[i] = stack_x[i-1] + test_vector[permutation[i]];
        stack_y[i] = stack_y[i-1] + 1 - test_vector[permutation[i]];
    }

    for(int i=0; i < predictions.size() - 1; i++) {
        auc += (stack_x(i+1) - stack_x(i)) * stack_y(i+1) / (NP*NN); //TODO:Make it Eigen
    }

    return auc;
}

template<typename YType>
void MF<YType>::init()
{
    assert(Yrows() > 0 && Ycols() > 0);
    mean_rating = Y.sum() / Y.nonZeros();
    U(0).resize(num_latent, Y.cols()); U(0).setZero();
    U(1).resize(num_latent, Y.rows()); U(1).setZero();
}

template<typename YType>
void MF<YType>::setRelationData(YType Y) {
    this->Y = Y;
    init();
}

template struct MF<SparseMatrix<double>>;
template struct MF<MatrixXd>;


//
//-- SparseMF specific stuff
//

void SparseMF::setRelationData(int* rows, int* cols, double* values, int N, int nrows, int ncols) {
    Y.resize(nrows, ncols);
    sparseFromIJV(Y, rows, cols, values, N);
    init();
}

void SparseMF::setRelationData(SparseDoubleMatrix &Y) {
    setRelationData(Y.rows, Y.cols, Y.vals, Y.nnz, Y.nrow, Y.ncol);
}

double SparseMF::var_total() const {
    double se = 0.0;

#pragma omp parallel for schedule(dynamic, 4) reduction(+:se)
    for (int k = 0; k < Y.outerSize(); ++k) {
        for (SparseMatrix<double>::InnerIterator it(Y,k); it; ++it) {
            se += square(it.value() - mean_rating);
        }
    }

    double var_total = se / Y.nonZeros();
    if (var_total <= 0.0 || std::isnan(var_total)) {
        // if var cannot be computed using 1.0
        var_total = 1.0;
    }

    return var_total;
}

double SparseMF::sumsq() const {
    double sumsq = 0.0;

#pragma omp parallel for schedule(dynamic, 4) reduction(+:sumsq)
    for (int j = 0; j < Y.outerSize(); j++) {
        auto Vj = col(1, j);
        for (SparseMatrix<double>::InnerIterator it(Y, j); it; ++it) {
            double Yhat = Vj.dot( col(0, it.row()) ) + mean_rating;
            sumsq += square(Yhat - it.value());
        }
    }

    return sumsq;
}

Factors::PnM SparseMF::get_pnm(int f, int n) {
    MatrixXd MM = MatrixXd::Zero(num_latent, num_latent);
    VectorXd rr = VectorXd::Zero(num_latent);

    for (SparseMatrix<double>::InnerIterator it(Yc.at(f), n); it; ++it) {
        auto col = V(f).col(it.row());
        rr.noalias() += col * it.value();
        MM.triangularView<Eigen::Lower>() += col * col.transpose();
    }

    return std::make_pair(rr, MM);
}

Factors::PnM SparseMF::get_probit_pnm(int f, int n)
{
    MatrixXd MM = MatrixXd::Zero(num_latent, num_latent);
    VectorXd rr = VectorXd::Zero(num_latent);

    auto u = U(f).col(n);
    for (SparseMatrix<double>::InnerIterator it(Yc.at(f), n); it; ++it) {
        auto col = V(f).col(it.row());
        MM.triangularView<Eigen::Lower>() += col * col.transpose();
        auto z = (2 * it.value() - 1) * fabs(col.dot(u) + bmrandn_single());
        rr.noalias() += col * z;
    }

    return std::make_pair(rr, MM);
}


//
//-- DenseMF specific stuff
//

// for the adaptive gaussian noise
double DenseMF::var_total() const {
    double se = (Y.array() - mean_rating).square().sum();

    double var_total = se / Y.nonZeros();
    if (var_total <= 0.0 || std::isnan(var_total)) {
        // if var cannot be computed using 1.0
        var_total = 1.0;
    }

    return var_total;
}

// for the adaptive gaussian noise
double DenseMF::sumsq() const {
    double sumsq = 0.0;

#pragma omp parallel for schedule(dynamic, 4) reduction(+:sumsq)
    for (int j = 0; j < Y.cols(); j++) {
        auto Vj = U(1).col(j);
        for (int i = 0; i < Y.rows(); i++) {
            double Yhat = Vj.dot( U(0).col(j) ) + mean_rating;
            sumsq += square(Yhat - Y(i,j));
        }
    }

    return sumsq;
}

Factors::PnM DenseMF::get_pnm(int f, int d) {
    auto &Y = Yc.at(f);
    VectorXd rr = (V(f) * Y.col(d));
    return std::make_pair(rr, VV.at(f));
}

void DenseMF::update_pnm(int f) {
    auto &Vf = V(f);
#pragma omp parallel for schedule(dynamic, 2) reduction(MatrixPlus:XX)
    for(int n = 0; n < Vf.cols(); n++) {
        auto v = Vf.col(n);
        VV.at(f) += v * v.transpose();
    }


}



