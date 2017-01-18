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

using namespace std; 
using namespace Eigen;

void SparseMF::setRelationData(int* rows, int* cols, double* values, int N, int nrows, int ncols) {
  Y.resize(nrows, ncols);
  sparseFromIJV(Y, rows, cols, values, N);
  init();
}

void Factors::setRelationDataTest(int* rows, int* cols, double* values, int N, int nrows, int ncols) {
  assert(nrows == Yrows() && ncols == Ycols() && 
         "Size of train must be equal to size of test");
    
  Ytest.resize(nrows, ncols);
  sparseFromIJV(Ytest, rows, cols, values, N);
  init();
}

void Factors::init()
{
  predictions     = VectorXd::Zero( Ytest.nonZeros() );
  predictions_var = VectorXd::Zero( Ytest.nonZeros() );
}

void DenseMF::init()
{
    assert(Yrows() > 0 && Ycols() > 0);
    mean_rating = Y.sum() / Y.nonZeros();
    U(0).resize(num_latent, Y.cols()); U(0).setZero();
    U(1).resize(num_latent, Y.rows()); U(1).setZero();
    Ut.at(0).resize(num_latent, Y.cols()); Ut.at(0).setZero();
    Ut.at(1).resize(num_latent, Y.rows()); Ut.at(1).setZero();
}

void SparseMF::init()
{
    assert(Yrows() > 0 && Ycols() > 0);
    mean_rating = Y.sum() / Y.nonZeros();
    U(0).resize(num_latent, Y.cols()); U(0).setZero();
    U(1).resize(num_latent, Y.rows()); U(1).setZero();
}

void SparseMF::setRelationData(SparseDoubleMatrix &Y) {
   setRelationData(Y.rows, Y.cols, Y.vals, Y.nnz, Y.nrow, Y.ncol);
}
   
void DenseMF::setRelationData(MatrixXd Y) {
    this->Y = Y;
    init();
}
 
void Factors::setRelationDataTest(SparseDoubleMatrix &Y) {
   setRelationDataTest(Y.rows, Y.cols, Y.vals, Y.nnz, Y.nrow, Y.ncol);
}
 
void Factors::setRelationDataTest(SparseMatrixD Y) {
    Ytest = Y;
    init();
}

Eigen::VectorXd Factors::getStds() {
  VectorXd std(Ytest.nonZeros());
  const int n = std.size();
  const double inorm = 1.0 / (iter - 1);
#pragma omp parallel for schedule(static)
  for (int i = 0; i < n; i++) {
    std[i] = sqrt(predictions_var[i] * inorm);
  }
  return std;
}

// assumes matrix (not tensor)
Eigen::MatrixXd Factors::getTestData() {
    MatrixXd coords(Ytest.nonZeros(), 3);
#pragma omp parallel for schedule(dynamic, 2)
    for (int k = 0; k < Ytest.outerSize(); ++k) {
        int idx = Ytest.outerIndexPtr()[k];
        for (SparseMatrix<double>::InnerIterator it(Ytest,k); it; ++it) {
            coords(idx, 0) = it.row();
            coords(idx, 1) = it.col();
            coords(idx, 2) = it.value();
            idx++;
        }
    }
    return coords;
}

//--- output model to files

void Factors::savePredictions(std::string save_prefix) {
    VectorXd yhat_sd_raw  = getStds();
    MatrixXd testdata_raw = getTestData();

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

void Factors::saveModel(std::string save_prefix) {
    int i = 0;
    for(auto &U : factors) {
        writeToCSVfile(save_prefix + "U" + std::to_string(++i) + "-latents.csv", U);
    }
    savePredictions(save_prefix);
}

///--- update RMSE and AUC

void Factors::update_rmse(bool burnin)
{
    if (Ytest.nonZeros() == 0) return;

    double se = 0.0, se_avg = 0.0;
#pragma omp parallel for schedule(dynamic,8) reduction(+:se, se_avg)
    for (int k = 0; k < Ytest.outerSize(); ++k) {
        int idx = Ytest.outerIndexPtr()[k];
        for (Eigen::SparseMatrix<double>::InnerIterator it(Ytest,k); it; ++it) {
            const double pred = col(0,it.col()).dot(col(1,it.row())) + mean_rating;
            se += square(it.value() - pred);

            // https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online_algorithm
            double pred_avg;
            if (burnin) {
                pred_avg = pred;
            } else {
                double delta = pred - predictions[idx];
                pred_avg = (predictions[idx] + delta / (iter + 1));
                predictions_var[idx] += delta * (pred - pred_avg);
            }
            se_avg += square(it.value() - pred_avg);
            predictions[idx++] = pred_avg;
        }
    }

    const unsigned N = Ytest.nonZeros();
    rmse = sqrt( se / N );
    rmse_avg = sqrt( se_avg / N );
    if (!burnin) iter++;
}

void Factors::update_auc(bool burnin)
{
    if (Ytest.nonZeros() == 0) return;

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
}
