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

#include "macau.h"
#include "mvnormal.h"
#include "utils.h"
#include "latentprior.h"

using namespace std; 
using namespace Eigen;

static volatile bool keepRunning = true;

void intHandler(int dummy) {
  keepRunning = false;
  printf("[Received Ctrl-C. Stopping after finishing the current iteration.]\n");
}

FixedGaussianNoise &Macau::setPrecision(double p) {
  FixedGaussianNoise *n = new FixedGaussianNoise(model, p);
  noise.reset(n);
  return *n;
}

AdaptiveGaussianNoise &Macau::setAdaptivePrecision(double sn_init, double sn_max) {
  AdaptiveGaussianNoise *n = new AdaptiveGaussianNoise(model, sn_init, sn_max);
  noise.reset(n);
  return *n;
}

ProbitNoise &Macau::setProbit() {
  ProbitNoise *n = new ProbitNoise(model);
  noise.reset(n);
  return *n;
}

void Macau::setSamples(int b, int n) {
  burnin = b;
  this->nsamples = n;
}

void SparseMF::setRelationData(int* rows, int* cols, double* values, int N, int nrows, int ncols) {
  Y.resize(nrows, ncols);
  sparseFromIJV(Y, rows, cols, values, N);
  mean_rating = Y.sum() / Y.nonZeros();
}

void SparseMF::setRelationDataTest(int* rows, int* cols, double* values, int N, int nrows, int ncols) {
  assert(nrows == Y.rows() && ncols == Y.cols() && 
         "Size of train must be equal to size of test");
    
  Ytest.resize(nrows, ncols);
  sparseFromIJV(Ytest, rows, cols, values, N);
}

void SparseMF::setRelationData(SparseDoubleMatrix &Y) {
   setRelationData(Y.rows, Y.cols, Y.vals, Y.nnz, Y.nrow, Y.ncol);
}
    
void SparseMF::setRelationDataTest(SparseDoubleMatrix &Y) {
   setRelationDataTest(Y.rows, Y.cols, Y.vals, Y.nnz, Y.nrow, Y.ncol);
}

double Macau::getRmseTest() { return rmse_test; }

void SparseMF::init() {
    U(0).resize(num_latent, Y.cols()); U(0).setZero();
    U(1).resize(num_latent, Y.rows()); U(1).setZero();
    predictions     = VectorXd::Zero( Ytest.nonZeros() );
    predictions_var = VectorXd::Zero( Ytest.nonZeros() );
}

void Macau::init() {
  unsigned seed1 = std::chrono::system_clock::now().time_since_epoch().count();
  if (priors.size() != 2) {
    throw std::runtime_error("Only 2 priors are supported.");
  }
  init_bmrng(seed1);
  model.init();
  noise->init();
  keepRunning = true;
}

Macau::~Macau() {}

inline double sqr(double x) { return x*x; }

void Macau::run() {
    init();
    if (verbose) {
        std::cout << noise->getInitStatus() << endl;
        std::cout << "Sampling" << endl;
    }
    if (save_model) {
        saveGlobalParams();
    }
    signal(SIGINT, intHandler);

    const int num_rows = model.Y.rows();
    const int num_cols = model.Y.cols();

    auto start = tick();
    for (int i = 0; i < burnin + nsamples; i++) {
        if (keepRunning == false) {
            keepRunning = true;
            break;
        }
        if (verbose && i == burnin) {
            printf(" ====== Burn-in complete, averaging samples ====== \n");
        }
        auto starti = tick();

        // sample latent vectors
        for(auto &p : priors) p->sample_latents();

        // Sample hyperparams
        for(auto &p : priors) p->update_prior();
        noise->update();
        noise->evalModel(i < burnin);

        auto endi = tick();
        auto elapsed = endi - start;
        double samples_per_sec = (i + 1) * (num_rows + num_cols) / elapsed;
        double elapsedi = endi - starti;

        saveModel(i - burnin + 1);
        printStatus(i, elapsedi, samples_per_sec);
        rmse_test = noise->getEvalMetric();
    }
}

void Macau::printStatus(int i, double elapsedi, double samples_per_sec) {
  if(!verbose) return;
  double norm0 = priors[0]->getLinkNorm();
  double norm1 = priors[1]->getLinkNorm();
  double snorm0 = model.U(0).norm();
  double snorm1 = model.U(1).norm();
  printf("Iter %3d/%3d: %s  U:[%1.2e, %1.2e]  Side:[%1.2e, %1.2e] %s [took %0.1fs]\n", i, burnin + nsamples,
          noise->getEvalString().c_str(),
          snorm0, snorm1, norm0, norm1, noise->getStatus().c_str(), elapsedi);
}

Eigen::VectorXd SparseMF::getStds(int iter) {
  VectorXd std(Ytest.nonZeros());
  if (iter <= 1) {
    std.setConstant(NAN);
    return std;
  }
  const int n = std.size();
  const double inorm = 1.0 / (iter - 1);
#pragma omp parallel for schedule(static)
  for (int i = 0; i < n; i++) {
    std[i] = sqrt(predictions_var[i] * inorm);
  }
  return std;
}

// assumes matrix (not tensor)
Eigen::MatrixXd SparseMF::getTestData() {
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

void Macau::saveModel(int isample) {
    if (!save_model || isample < 0) return;
    string fprefix = save_prefix + "-sample" + std::to_string(isample) + "-";
    // saving latent matrices
    for (unsigned int i = 0; i < model.factors.size(); i++) {
        writeToCSVfile(fprefix + "U" + std::to_string(i+1) + "-latents.csv", model.U(i));
        priors[i]->savePriorInfo(fprefix + "U" + std::to_string(i+1));
    }
    savePredictions();
}

void Macau::savePredictions() {
    VectorXd yhat_raw     = getPredictions();
    VectorXd yhat_sd_raw  = model.getStds(nsamples);
    MatrixXd testdata_raw = model.getTestData();

    std::string fname_pred = save_prefix + "-predictions.csv";
    std::ofstream predfile;
    predfile.open(fname_pred);
    predfile << "row,col,y,y_pred,y_pred_std\n";
    for (int i = 0; i < yhat_raw.size(); i++) {
        predfile << to_string( (int)testdata_raw(i,0) );
        predfile << "," << to_string( (int)testdata_raw(i,1) );
        predfile << "," << to_string( testdata_raw(i,2) );
        predfile << "," << to_string( yhat_raw(i) );
        predfile << "," << to_string( yhat_sd_raw(i) );
        predfile << "\n";
    }
    predfile.close();
    printf("Saved predictions into '%s'.\n", fname_pred.c_str());

}

void Macau::saveGlobalParams() {
  VectorXd means(1);
  means << model.mean_rating;
  writeToCSVfile(save_prefix + "-meanvalue.csv", means);
}

void SparseMF::update_rmse(bool burnin)
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

void SparseMF::update_auc(bool burnin)
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
