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

#include "utils.h"
#include "model.h"
#include "mvnormal.h"

using namespace std; 
using namespace Eigen;

int global_num_latent = -1;

void Factors::setRelationDataTest(int* rows, int* cols, double* values, int N, int nrows, int ncols) {
    Ytest.resize(nrows, ncols);
    sparseFromIJV(Ytest, rows, cols, values, N);
}
 
void Factors::setRelationDataTest(SparseDoubleMatrix &Y) {
    Ytest = to_eigen(Y);
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

std::ostream &Factors::printInitStatus(std::ostream &os, std::string indent)
{
    os << indent << "Type: " << name << "\n";
    os << indent << "Num-latents: " << num_latent << "\n";
    os << indent << "Train data: " << Ynnz() << " [" << Yrows() << " x " << Ycols() << "]\n";
    os << indent << "Test data: " << Ytest.nonZeros() << " [" << Ytest.rows() << " x " << Ytest.cols() << "]\n";
    return os;
}


template<typename YType>
void MF<YType>::init_base()
{
    assert(Yrows() > 0 && Ycols() > 0);
    assert(Ytest.rows() == Yrows() && Ytest.cols() == Ycols() && "Size of train must be equal to size of test");

    mean_rating = Y.sum() / Y.nonZeros();

    U(0) = nrandn(num_latent, Y.cols());
    U(1) = nrandn(num_latent, Y.rows());

    Yc.push_back(Y);
    Yc.push_back(Y.transpose());
}

template<>
void MF<SparseMatrixD>::init()
{
    init_base();
    Yc.at(0).coeffs() -= mean_rating;
    Yc.at(1).coeffs() -= mean_rating;
    name = "Sparse" + name;
}


template<>
void MF<Eigen::MatrixXd>::init()
{
    init_base();
    Yc.at(0).array() -= this->mean_rating;
    Yc.at(1).array() -= this->mean_rating;
    name = "Dense" + name;
}

template<typename YType>
void MF<YType>::setRelationData(YType Y) {
    this->Y = Y;
}

template struct MF<SparseMatrix<double>>;
template struct MF<MatrixXd>;


template<>
double MF<Eigen::MatrixXd>::var_total() const {
    auto &Y = Yc.at(0);
    double se = Y.array().square().sum();

    double var_total = se / Y.nonZeros();
    if (var_total <= 0.0 || std::isnan(var_total)) {
        // if var cannot be computed using 1.0
        var_total = 1.0;
    }

    return var_total;
}

template<>
double MF<SparseMatrixD>::var_total() const {
    double se = 0.0;
    auto &Y = Yc.at(0);

#pragma omp parallel for schedule(dynamic, 4) reduction(+:se)
    for (int k = 0; k < Y.outerSize(); ++k) {
        for (SparseMatrix<double>::InnerIterator it(Y,k); it; ++it) {
            se += square(it.value());
        }
    }

    double var_total = se / Y.nonZeros();
    if (var_total <= 0.0 || std::isnan(var_total)) {
        // if var cannot be computed using 1.0
        var_total = 1.0;
    }

    return var_total;
}

// for the adaptive gaussian noise
template<>
double MF<Eigen::MatrixXd>::sumsq() const {
    double sumsq = 0.0;

#pragma omp parallel for schedule(dynamic, 4) reduction(+:sumsq)
    for (int j = 0; j < this->Y.cols(); j++) {
        auto Vj = this->U(1).col(j);
        for (int i = 0; i < this->Y.rows(); i++) {
            double Yhat = Vj.dot( this->U(0).col(j) ) + this->mean_rating;
            sumsq += square(Yhat - this->Y(i,j));
        }
    }

    return sumsq;
}

template<>
double MF<SparseMatrixD>::sumsq() const {
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

template<>
void MF<SparseMatrixD>::setRelationData(int* rows, int* cols, double* values, int N, int nrows, int ncols) {
    Y.resize(nrows, ncols);
    sparseFromIJV(Y, rows, cols, values, N);
}

template<>
void MF<SparseMatrixD>::setRelationData(SparseDoubleMatrix &Y) {
    setRelationData(Y.rows, Y.cols, Y.vals, Y.nnz, Y.nrow, Y.ncol);
}

//
//-- SparseMF specific stuff
//

Factors::PnM SparseMF::get_pnm(int f, int n) {
    MatrixXd MM = MatrixXd::Zero(num_latent, num_latent);
    VectorXd rr = VectorXd::Zero(num_latent);
    auto &Y = Yc.at(f);
    MatrixXd &Vf = V(f);
    const int C = Y.col(n).nonZeros();
    if (C > 1e6) {
        auto from = Y.outerIndexPtr()[n];
        auto to = Y.outerIndexPtr()[n+1];
#pragma omp parallel for schedule(dynamic, 20) reduction(VectorPlus:rr) reduction(MatrixPlus:MM)
        for(int i=from; i<to; ++i) {
            auto val = Y.valuePtr()[i];
            auto idx = Y.innerIndexPtr()[i];
            const auto &col = Vf.col(idx);
            rr.noalias() += col * val;
            MM += col * col.transpose();
        }
    } else {
        for (SparseMatrix<double>::InnerIterator it(Y, n); it; ++it) {
            const auto &col = Vf.col(it.row());
            rr.noalias() += col * it.value();
            MM += col * col.transpose();
        }
    }

    return std::make_pair(rr, MM);
}

Factors::PnM SparseMF::get_probit_pnm(int f, int n)
{
    MatrixXd MM = MatrixXd::Zero(num_latent, num_latent);
    VectorXd rr = VectorXd::Zero(num_latent);

    auto u = U(f).col(n);
    for (SparseMatrix<double>::InnerIterator it(Yc.at(f), n); it; ++it) {
        const auto &col = V(f).col(it.row());
        MM += col * col.transpose();
        auto z = (2 * it.value() - 1) * fabs(col.dot(u) + bmrandn_single());
        rr.noalias() += col * z;
    }

    return std::make_pair(rr, MM);
}


//
//-- DenseMF specific stuff
//

template<class YType>
Factors::PnM DenseMF<YType>::get_pnm(int f, int d) {
    auto &Y = this->Yc.at(f);
    VectorXd rr = (this->V(f) * Y.col(d));
    return std::make_pair(rr, VV.at(f));
}

template<class YType>
void DenseMF<YType>::update_pnm(int f) {
    auto &Vf = this->V(f);
    auto &VVf = VV.at(f);
    VVf = MatrixXd::Zero(this->num_latent, this->num_latent);

#pragma omp parallel for schedule(dynamic, 2) reduction(MatrixPlus:VVf)
    for(int n = 0; n < Vf.cols(); n++) {
        auto v = Vf.col(n);
        VVf += v * v.transpose();
    }
}

template struct DenseMF<Eigen::MatrixXd>;
template struct DenseMF<SparseMatrixD>;
