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

#include <getopt.h>
#include <signal.h>

#include "macau.h"
#include "mvnormal.h"
#include "utils.h"
#include "latentprior.h"
#include "macauoneprior.h"
#include "omp_util.h"
#include "linop.h"

using namespace std; 
using namespace Eigen;

extern "C" {
  #include <dsparse.h>
}

//-- add model
SparseMF &Macau::sparseModel(int num_latent) {
  SparseMF *n = new SparseMF(num_latent);
  model.reset(n);
  return *n;
}

DenseMF  &Macau::denseModel(int num_latent) {
  DenseMF *n = new DenseMF(num_latent);
  model.reset(n);
  return *n;
}

FixedGaussianNoise &Macau::setPrecision(double p) {
  FixedGaussianNoise *n = new FixedGaussianNoise(*model, p);
  noise.reset(n);
  return *n;
}

AdaptiveGaussianNoise &Macau::setAdaptivePrecision(double sn_init, double sn_max) {
  AdaptiveGaussianNoise *n = new AdaptiveGaussianNoise(*model, sn_init, sn_max);
  noise.reset(n);
  return *n;
}

ProbitNoise &Macau::setProbit() {
  ProbitNoise *n = new ProbitNoise(*model);
  noise.reset(n);
  return *n;
}

void Macau::setSamples(int b, int n) {
  burnin = b;
  nsamples = n;
}

void Macau::init() {
  unsigned seed1 = std::chrono::system_clock::now().time_since_epoch().count();
  if (priors.size() != 2) {
    throw std::runtime_error("Only 2 priors are supported.");
  }
  init_bmrng(seed1);
  noise->init();
  if (verbose) {
      std::cout << noise->getInitStatus() << endl;
      std::cout << "Sampling" << endl;
  }
  if (save_model) model->saveGlobalParams(save_prefix);
  iter = 0;
}

void Macau::run() {
    init();
    while (iter < burnin + nsamples) {
        step();
        iter++;
    }
}


void Macau::step() {
    if (verbose && iter == burnin) {
        printf(" ====== Burn-in complete, averaging samples ====== \n");
    }
    auto starti = tick();

    // Sample hyperparams + latents
    for(auto &p : priors) p->pre_update();
    for(auto &p : priors) p->sample_latents();
    for(auto &p : priors) p->post_update();

    noise->update();

    auto endi = tick();

    saveModel(iter - burnin + 1);
    printStatus(endi - starti);
}

volatile bool PythonMacau::keepRunning;

void PythonMacau::run() {
    keepRunning = true;
    signal(SIGINT, intHandler);
    for (iter = 0; iter < burnin + nsamples; iter++) {
        if (keepRunning == false) {
            keepRunning = true;
            break;
        }
        step();
    }
}

void PythonMacau::intHandler(int) {
  keepRunning = false;
  printf("[Received Ctrl-C. Stopping after finishing the current iteration.]\n");
}

void Macau::setFromArgs(int argc, char** argv, bool print) {
    char* fname_train         = NULL;
    char* fname_test          = NULL;
    char* fname_row_features  = NULL;
    std::string output_prefix = std::string("result");
    double precision          = 5.0;
    double lambda_beta        = 10.0;
    double tol                = 1e-6;
    int burnin                = 200;
    int nsamples              = 800;
    int num_latent            = 96;

    auto die = [print](std::string message) {
        if (print) std::cout << message;
        exit(-1);
    };

    std::string usage = 
        "Usage:\n"
        "  macau_mpi --train <train_file> --row-features <feature-file> [options]\n"
        "Optional:\n"
        "  --test    test_file  test data (for computing RMSE)\n"
        "  --burnin        200  number of samples to discard\n"
        "  --nsamples      800  number of samples to collect\n"
        "  --num-latent     96  number of latent dimensions\n"
        "  --precision     5.0  precision of observations\n"
        "  --lambda-beta  10.0  initial value of lambda beta\n"
        "  --tol          1e-6  tolerance for CG\n"
        "  --output    results  prefix for result files\n\n";

    // reading command line arguments
    while (1) {
        static struct option long_options[] =
        {
            {"train",      required_argument, 0, 't'},
            {"test",       required_argument, 0, 'e'},
            {"row-features", required_argument, 0, 'r'},
            {"precision",  required_argument, 0, 'p'},
            {"burnin",     required_argument, 0, 'b'},
            {"nsamples",   required_argument, 0, 'n'},
            {"output",     required_argument, 0, 'o'},
            {"num-latent", required_argument, 0, 'l'},
            {"lambda-beta",required_argument, 0, 'a'},
            {"tol",        required_argument, 0, 'c'},
            {0, 0, 0, 0}
        };
        int option_index = 0;
        int c = getopt_long(argc, argv, "t:e:r:p:b:n:o:a:c:", long_options, &option_index);
        if (c == -1)
            break;

        switch (c) {
            case 'a': lambda_beta   = strtod(optarg, NULL); break;
            case 'b': burnin        = strtol(optarg, NULL, 10); break;
            case 'c': tol           = atof(optarg); break;
            case 'e': fname_test    = optarg; break;
            case 'l': num_latent    = strtol(optarg, NULL, 10); break;
            case 'n': nsamples      = strtol(optarg, NULL, 10); break;
            case 'o': output_prefix = std::string(optarg); break;
            case 'p': precision     = strtod(optarg, NULL); break;
            case 'r': fname_row_features = optarg; break;
            case 't': fname_train = optarg; break;
            case '?':
            default : die(usage);
        }
    }
    if (fname_train == NULL || fname_row_features == NULL) {
        die(usage + "[ERROR]\nMissing parameters '--matrix' or '--row-features'.\n");
    }

    if (print) {
        printf("Train data:    '%s'\n", fname_train);
        printf("Test data:     '%s'\n", fname_test==NULL ?"" :fname_test);
        printf("Row features:  '%s'\n", fname_row_features);
        printf("Output prefix: '%s'\n", output_prefix.c_str());
        printf("Burn-in:       %d\n", burnin);
        printf("Samples:       %d\n", nsamples);
        printf("Num-latents:   %d\n", num_latent);
        printf("Precision:     %.1f\n", precision);
        printf("Lambda-beta:   %.1f\n", lambda_beta);
        printf("tol:           %.1e\n", tol);
    }
    if ( ! file_exists(fname_train) ) {
        die(std::string("[ERROR]\nTrain data file '") + fname_train + "' not found.\n");
    }
    if ( ! file_exists(fname_row_features) ) {
        die(std::string("[ERROR]\nRow feature file '") + fname_row_features + "' not found.\n");
    }
    if ( (fname_test != NULL) && ! file_exists(fname_test) ) {
        die(std::string("[ERROR]\nTest data file '") + fname_test + "' not found.\n");
    }

    // Step 1. Loading data
    //std::unique_ptr<SparseFeat> row_features = load_bcsr(fname_row_features);
    auto row_features = load_bcsr(fname_row_features);
    if (print) {
        printf("Row features:   [%d x %d].\n", row_features->rows(), row_features->cols());
    }
    SparseDoubleMatrix* Y     = NULL;
    SparseDoubleMatrix* Ytest = NULL;

    Macau macau;
    SparseMF& model = sparseModel(num_latent);
    setSamples(burnin, nsamples);

    // -- noise model + general parameters
    setPrecision(precision);

    setVerbose(true);
    Y = read_sdm(fname_train);
    model.setRelationData(*Y);

    //-- Normal column prior
    //addPrior<SparseNormalPrior>();
    addPrior<SparseSpikeAndSlabPrior>();

    //-- row prior with side information
    auto &prior_u = addPrior<MacauOnePrior<SparseFeat>>();
    prior_u.addSideInfo(row_features, false);
    prior_u.setLambdaBeta(lambda_beta);
    //prior_u.setTol(tol);

    // test data
    if (fname_test != NULL) {
        Ytest = read_sdm(fname_test);
        model.setRelationDataTest(*Ytest);
    }

    if (print) {
        printf("Training data:  %ld [%d x %d]\n", Y->nnz, Y->nrow, Y->ncol);
        if (Ytest != NULL) {
            printf("Test data:      %ld [%d x %d]\n", Ytest->nnz, Ytest->nrow, Ytest->ncol);
        } else {
            printf("Test data:      --\n");
        }
    }

    delete Y;
    if (Ytest) delete Ytest;
}


void Macau::printStatus(double elapsedi) {
    if(!verbose) return;
    double norm0 = priors[0]->getLinkNorm();
    double norm1 = priors[1]->getLinkNorm();

    double snorm0 = model->U(0).norm();
    double snorm1 = model->U(1).norm();

    std::pair<double,double> rmse_test = model->getRMSE(iter, burnin);
    double auc = model->auc();

    printf("Iter %3d/%3d: RMSE: %.4f (1samp: %.4f)  AUC: %.4f U:[%1.2e, %1.2e]  Side:[%1.2e, %1.2e] %s [took %0.1fs]\n",
            iter, burnin + nsamples, rmse_test.second, rmse_test.first, auc,
            snorm0, snorm1, norm0, norm1, noise->getStatus().c_str(), elapsedi);
}

void Macau::saveModel(int isample) {
    if (!save_model || isample < 0) return;
    string fprefix = save_prefix + "-sample" + std::to_string(isample) + "-";
    model->saveModel(fprefix, isample, burnin);
    for(auto &p : priors) p->savePriorInfo(fprefix);
}

