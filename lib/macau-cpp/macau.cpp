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

//-- add model
SparseMF &MacauBase::sparseModel(int num_latent) {
  SparseMF *n = new SparseMF(num_latent);
  model.reset(n);
  return *n;
}

DenseMF  &MacauBase::denseModel(int num_latent) {
  DenseMF *n = new DenseMF(num_latent);
  model.reset(n);
  return *n;
}

FixedGaussianNoise &MacauBase::setPrecision(double p) {
  FixedGaussianNoise *n = new FixedGaussianNoise(*model, p);
  noise.reset(n);
  return *n;
}

AdaptiveGaussianNoise &MacauBase::setAdaptivePrecision(double sn_init, double sn_max) {
  AdaptiveGaussianNoise *n = new AdaptiveGaussianNoise(*model, sn_init, sn_max);
  noise.reset(n);
  return *n;
}

void MacauBase::init() {
  if (priors.size() != 2) throw std::runtime_error("Only 2 priors are supported.");
  for( auto &p : priors) p->init();
  model->init();
  noise->init();
}

void MacauBase::step() {
    for(auto &p : priors) p->sample_latents();
    noise->update();
}


//--- 

void Macau::setSamples(int b, int n) {
  burnin = b;
  nsamples = n;
}

void Macau::init() {
  MacauBase::init();
  unsigned seed1 = std::chrono::system_clock::now().time_since_epoch().count();
  init_bmrng(seed1);
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
    MacauBase::step();
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

// 
//-- cmdline handling stuff
//
template<class SideInfo>
inline void addMacauPrior(Macau &m, std::string prior_name, unique_ptr<SideInfo> &features, double lambda_beta, double tol)
{
    if(prior_name == "macau" || prior_name == "default"){
        auto &prior = m.addPrior<MacauPrior<SideInfo>>();
        prior.addSideInfo(features, false);
        prior.setLambdaBeta(lambda_beta);
        prior.setTol(tol);
    } else if(prior_name == "macauone") {
        auto &prior = m.addPrior<MacauOnePrior<SideInfo>>();
        prior.addSideInfo(features, false);
        prior.setLambdaBeta(lambda_beta);
    } else {
        throw std::runtime_error("Unknown prior with side info: " + prior_name);
    }
}

void add_prior(Macau &macau, std::string prior_name, std::string fname_features, double lambda_beta, double tol)
{
    //-- row prior with side information
    if (fname_features.size()) {
        if (fname_features.find(".sdm") != std::string::npos) {
            auto row_features = std::unique_ptr<SparseDoubleFeat>(load_csr(fname_features.c_str()));
            addMacauPrior(macau, prior_name, row_features, lambda_beta, tol);
        } else if (fname_features.find(".sbm") != std::string::npos) {
            auto features = load_bcsr(fname_features.c_str());
            addMacauPrior(macau, prior_name, features, lambda_beta, tol);
        } else {
            throw std::runtime_error("Train row_features file: expecing .sdm or .sbm, got " + std::string(fname_features));
        }
    } else if(prior_name == "normal" || prior_name == "default") {
        macau.addPrior<NormalPrior>();
    } else if(prior_name == "spikeandslab") {
        macau.addPrior<SpikeAndSlabPrior>();
    } else {
        throw std::runtime_error("Unknown prior without side info: " + prior_name);
    }
}

void Macau::setFromArgs(int argc, char** argv, bool print) {
    std::string fname_train;
    std::string fname_test;
    std::string fname_row_features;
    std::string fname_col_features;
    std::string row_prior("default");
    std::string col_prior("default");
    std::string output_prefix("result");

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
        "  macau_mpi --train <train_file> <feature-file> [options]\n"
        "Optional:\n"
        "  --row-prior <normal|spikeandslab|macau|macauone>\n"
        "  --col-prior <normal|spikeandslab|macau|macauone>\n\n"
        "  --row-features file  side info for rows\n"
        "  --col-features file  side info for cols\n"
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
            {"train",        required_argument, 0, 't'},
            {"test",         required_argument, 0, 'e'},
            {"row-features", required_argument, 0, 'r'},
            {"col-features", required_argument, 0, 'f'},
            {"row-prior",    required_argument, 0, 'q'},
            {"col-prior",    required_argument, 0, 's'},
            {"precision",    required_argument, 0, 'p'},
            {"burnin",       required_argument, 0, 'b'},
            {"nsamples",     required_argument, 0, 'n'},
            {"output",       required_argument, 0, 'o'},
            {"num-latent",   required_argument, 0, 'l'},
            {"lambda-beta",  required_argument, 0, 'a'},
            {"tol",          required_argument, 0, 'c'},
            {0, 0, 0, 0}
        };
        int option_index = 0;
        int c = getopt_long(argc, argv, "t:e:r:f:p:b:n:o:a:c:", long_options, &option_index);
        if (c == -1)
            break;

        switch (c) {
            case 'a': lambda_beta        = strtod(optarg, NULL); break;
            case 'b': burnin             = strtol(optarg, NULL, 10); break;
            case 'c': tol                = atof(optarg); break;
            case 'e': fname_test         = optarg; break;
            case 'l': num_latent         = strtol(optarg, NULL, 10); break;
            case 'n': nsamples           = strtol(optarg, NULL, 10); break;
            case 'o': output_prefix      = std::string(optarg); break;
            case 'p': precision          = strtod(optarg, NULL); break;
            case 'r': fname_row_features = optarg; break;
            case 'f': fname_col_features = optarg; break;
            case 'q': row_prior          = optarg; break;
            case 's': col_prior          = optarg; break;
            case 't': fname_train        = optarg; break;
            case '?':
            default : die(usage);
        }
    }
    if (fname_train.empty()) {
        die(usage + "[ERROR]\nMissing parameters '--train'\n");
    }

    if (print) {
        printf("Train data:    '%s'\n", fname_train.c_str());
        printf("Test data:     '%s'\n", fname_test.c_str());
        printf("Row prior:     '%s'\n", row_prior.c_str());
        printf("Col prior:     '%s'\n", col_prior.c_str());
        printf("Row features:  '%s'\n", fname_row_features.c_str());
        printf("Col features:  '%s'\n", fname_col_features.c_str());
        printf("Output prefix: '%s'\n", output_prefix.c_str());
        printf("Burn-in:       %d\n",   burnin);
        printf("Samples:       %d\n",   nsamples);
        printf("Num-latents:   %d\n",   num_latent);
        printf("Precision:     %.1f\n", precision);
        printf("Lambda-beta:   %.1f\n", lambda_beta);
        printf("tol:           %.1e\n", tol);
    }
    if ( ! file_exists(fname_train) ) {
        die(std::string("[ERROR]\nTrain data file '") + fname_train + "' not found.\n");
    }
    if ( fname_row_features.size() && ! file_exists(fname_row_features) ) {
        die(std::string("[ERROR]\nRow feature file '") + fname_row_features + "' not found.\n");
    }
    if ( fname_col_features.size() && ! file_exists(fname_col_features) ) {
        die(std::string("[ERROR]\nCol feature file '") + fname_col_features + "' not found.\n");
    }
    if ( fname_test.size() && ! file_exists(fname_test) ) {
        die(std::string("[ERROR]\nTest data file '") + fname_test + "' not found.\n");
    }

    // Load main Y matrix file
    if (fname_train.find(".sdm") != std::string::npos) {
        SparseMF& model = sparseModel(num_latent);
        SparseDoubleMatrix* Y = read_sdm(fname_train.c_str());
        model.setRelationData(*Y);
        delete Y;
    } else if (fname_train.find(".ddm") != std::string::npos) {
        DenseMF& model = denseModel(num_latent);
        MatrixXd Y = read_ddm<MatrixXd>(fname_train.c_str());
        model.setRelationData(Y);
    } else {
        die("Train data file: expecing .sdm or .ddm, got " + std::string(fname_train));
    }

    setSamples(burnin, nsamples);
    setPrecision(precision);
    setVerbose(true);

    // test data
    if (fname_test.size()) {
        auto Ytest = read_sdm(fname_test.c_str());
        model->setRelationDataTest(*Ytest);
        delete Ytest;
    }


    add_prior(*this, col_prior, fname_col_features, lambda_beta, tol);
    add_prior(*this, row_prior, fname_row_features, lambda_beta, tol);

/*
    if (print) {
        printf("Row features:   [%d x %d].\n", row_features->rows(), row_features->cols());
        printf("Training data:  %d [%d x %d]\n", model->Ynnz(), model->Yrows(), model->Ycols());
        if (Ytest != NULL) {
            printf("Test data:      %ld [%d x %d]\n", Ytest->nnz, Ytest->nrow, Ytest->ncol);
        } else {
            printf("Test data:      --\n");
        }
    }
    */
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

