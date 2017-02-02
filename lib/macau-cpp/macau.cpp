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
#include "gen_random.h"

using namespace std; 
using namespace Eigen;

//-- add model
//
template<class Model>
Model &MacauBase::addModel(int num_latent) {
    Model *n = new Model(num_latent);
    model.reset(n);
    return *n;
}

SparseMF &MacauBase::sparseModel(int num_latent) {
    return addModel<SparseMF>(num_latent);
}

DenseDenseMF &MacauBase::denseDenseModel(int num_latent) {
    return addModel<DenseDenseMF>(num_latent);
}

SparseDenseMF &MacauBase::sparseDenseModel(int num_latent) {
    return addModel<SparseDenseMF>(num_latent);
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
  model->init();
  for( auto &p : priors) p->init();
  noise->init();
}

void MacauBase::step() {
    for(auto &p : priors) p->sample_latents();
    noise->update();
}

std::ostream &MacauBase::printInitStatus(std::ostream &os, std::string indent) {
    os << indent << name << " {\n";
    os << indent << "  Priors: {\n";
    for( auto &p : priors) p->printInitStatus(os, indent + "    ");
    os << indent << "  }\n";
    os << indent << "  Factors: {\n";
    model->printInitStatus(os, indent + "    ");
    os << indent << "  }\n";
    os << indent << "  Noise: ";
    noise->printInitStatus(os, "");
    return os;
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
      printInitStatus(std::cout, "");
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

std::ostream &Macau::printInitStatus(std::ostream &os, std::string indent) {
    MacauBase::printInitStatus(os, indent);
    os << indent << "  Samples: " << burnin << " + " << nsamples << "\n";
    os << indent << "  Output prefix: " << save_prefix << "\n";
    os << indent << "}\n";
    return os;
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

template<class Prior>
inline void addMaster(Macau &macau, const Eigen::MatrixXd &sideinfo)
{
    auto &master_prior = macau.addPrior<MasterPrior<Prior>>();
    auto &slave_model = master_prior.template addSlave<DenseDenseMF>();
    slave_model.setRelationData(sideinfo);
}

template<class Prior>
inline void addMaster(Macau &macau, const SparseMatrixD &sideinfo)
{
    auto &master_prior = macau.addPrior<MasterPrior<Prior>>();
    auto &slave_model = master_prior.template addSlave<SparseDenseMF>();
    slave_model.setRelationData(sideinfo);
}


template<class SideInfo>
inline void addMaster(Macau &macau,std::string prior_name, const SideInfo &features)
{
    if(prior_name == "normal" || prior_name == "default") {
        addMaster<NormalPrior>(macau, features);
    } else if(prior_name == "spikeandslab") {
        addMaster<SpikeAndSlabPrior>(macau, features);
    } else {
        throw std::runtime_error("Unknown prior with side info: " + prior_name);
    }
}

void add_prior(Macau &macau, std::string prior_name, std::string fname_features, double lambda_beta, double tol)
{
    //-- row prior with side information
    if (fname_features.size()) {
        if (prior_name == "macau" || prior_name == "macauone") {
            if (fname_features.find(".sdm") != std::string::npos) {
                auto row_features = std::unique_ptr<SparseDoubleFeat>(load_csr(fname_features.c_str()));
                addMacauPrior(macau, prior_name, row_features, lambda_beta, tol);
            } else if (fname_features.find(".sbm") != std::string::npos) {
                auto features = load_bcsr(fname_features.c_str());
                addMacauPrior(macau, prior_name, features, lambda_beta, tol);
            } else {
                throw std::runtime_error("Train row_features file: expecing .sdm or .sbm, got " + std::string(fname_features));
            }
        } else {
            if (fname_features.find(".sdm") != std::string::npos) {
                auto features = read_sdm(fname_features.c_str());
                addMaster(macau, prior_name, to_eigen(*features));
                delete features;
            } else if (fname_features.find(".sbm") != std::string::npos) {
                auto features = read_sbm(fname_features.c_str());
                addMaster(macau, prior_name, to_eigen(*features));
                delete features;
            } else if (fname_features.find(".ddm") != std::string::npos) {
                auto features = read_ddm<Eigen::MatrixXd>(fname_features.c_str());
                addMaster(macau, prior_name, features);
            } else {
                throw std::runtime_error("Train features file: expecing .ddm, .sdm or .sbm, got " + std::string(fname_features));
            }
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
    double test_split         = .0;

    int burnin                = 200;
    int nsamples              = 800;
    int num_latent            = 96;

    auto die = [print](std::string message) {
        if (print) std::cout << message;
        exit(-1);
    };

    auto die_unless_file_exists = [&die](std::string fname) {
        if ( fname.size() && ! file_exists(fname) ) {
            die(std::string("[ERROR]\nFile '") + fname +  "' not found.\n");
        }
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
    die_unless_file_exists(fname_train);
    die_unless_file_exists(fname_row_features);
    die_unless_file_exists(fname_col_features);

    //-- check if fname_test is actually a number
    if ((test_split = atof(fname_test.c_str())) > .0) {
        fname_test.clear();
    }

    // Load main Y matrix file
    if (fname_train.find(".sdm") != std::string::npos) {
        SparseMF& model = sparseModel(num_latent);
        auto Ytrain = to_eigen(*read_sdm(fname_train.c_str()));
        if (test_split > .0) {
            auto Y = split(Ytrain, test_split);
            model.setRelationData(Y.first);
            model.setRelationDataTest(Y.second);
        } else {
            model.setRelationData(Ytrain);
        }
    } else if (fname_train.find(".ddm") != std::string::npos) {
        DenseDenseMF& model = denseDenseModel(num_latent);
        auto Ytrain = read_ddm<MatrixXd>(fname_train.c_str());
        if (test_split > .0) {
            auto Ytest = extract(Ytrain, test_split);
            model.setRelationDataTest(Ytest);
        }
        model.setRelationData(Ytrain);
    } else {
        die("Train data file: expecing .sdm or .ddm, got " + std::string(fname_train));
    }

    setSamples(burnin, nsamples);
    setPrecision(precision);
    setVerbose(true);

    // test data
    if (fname_test.size()) {
        die_unless_file_exists(fname_test);
        auto Ytest = read_sdm(fname_test.c_str());
        model->setRelationDataTest(*Ytest);
        delete Ytest;
    }

    add_prior(*this, col_prior, fname_col_features, lambda_beta, tol);
    add_prior(*this, row_prior, fname_row_features, lambda_beta, tol);
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

