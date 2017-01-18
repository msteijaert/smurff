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

//AdaptiveGaussianNoise &Macau::setAdaptivePrecision(double sn_init, double sn_max) {
//  AdaptiveGaussianNoise *n = new AdaptiveGaussianNoise(model, sn_init, sn_max);
//  noise.reset(n);
//  return *n;
//}
//
//ProbitNoise &Macau::setProbit() {
//  ProbitNoise *n = new ProbitNoise(model);
//  noise.reset(n);
//  return *n;
//}

void Macau::setSamples(int b, int n) {
  burnin = b;
  this->nsamples = n;
}
void Macau::init() {
  unsigned seed1 = std::chrono::system_clock::now().time_since_epoch().count();
  if (priors.size() != 2) {
    throw std::runtime_error("Only 2 priors are supported.");
  }
  init_bmrng(seed1);
  noise->init();
  keepRunning = true;
}

void Macau::run() {
    init();
    if (verbose) {
        std::cout << noise->getInitStatus() << endl;
        std::cout << "Sampling" << endl;
    }
    if (save_model) model.saveGlobalParams(save_prefix);
    signal(SIGINT, intHandler);

    const int num_rows = model.Yrows();
    const int num_cols = model.Ycols();

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

        // Sample hyperparams + latents
        for(auto &p : priors) p->pre_update();
        for(auto &p : priors) p->sample_latents();
        for(auto &p : priors) p->post_update();

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

void Macau::saveModel(int isample) {
    if (!save_model || isample < 0) return;
    string fprefix = save_prefix + "-sample" + std::to_string(isample) + "-";
    model.saveModel(fprefix);
    for(auto &p : priors) p->savePriorInfo(fprefix);
}

