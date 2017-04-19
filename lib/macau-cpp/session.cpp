
#include <iostream>
#include <cassert>
#include <fstream>
#include <string>
#include <algorithm>
#include <random>
#include <cstring>
#include <chrono>
#include <memory>
#include <cmath>
#include <argp.h>

#include <unsupported/Eigen/SparseExtra>
#include <Eigen/Sparse>

#include <getopt.h>
#include <signal.h>

#include "session.h"
#include "mvnormal.h"
#include "utils.h"
#include "latentprior.h"
#include "macauoneprior.h"
#include "omp_util.h"
#include "linop.h"
#include "gen_random.h"

using namespace std; 
using namespace Eigen;

namespace Macau {

//-- add model
//
template<class Model>
Model &BaseSession::addModel(int num_latent) {
    Model *n = new Model(num_latent);
    model.reset(n);
    return *n;
}

SparseMF &BaseSession::sparseModel(int num_latent) {
    return addModel<SparseMF>(num_latent);
}

SparseBinaryMF &BaseSession::sparseBinaryModel(int num_latent) {
    return addModel<SparseBinaryMF>(num_latent);
}

DenseDenseMF &BaseSession::denseDenseModel(int num_latent) {
    return addModel<DenseDenseMF>(num_latent);
}

SparseDenseMF &BaseSession::sparseDenseModel(int num_latent) {
    return addModel<SparseDenseMF>(num_latent);
}

FixedGaussianNoise &BaseSession::setPrecision(double p) {
  FixedGaussianNoise *n = new FixedGaussianNoise(*model, p);
  noise.reset(n);
  return *n;
}

AdaptiveGaussianNoise &BaseSession::setAdaptivePrecision(double sn_init, double sn_max) {
  AdaptiveGaussianNoise *n = new AdaptiveGaussianNoise(*model, sn_init, sn_max);
  noise.reset(n);
  return *n;
}

void BaseSession::init() {
    if (priors.size() != 2) throw std::runtime_error("Only 2 priors are supported (have" + std::to_string(priors.size()) + ").");
    model->init();
    for( auto &p : priors) p->init();
    noise->init();
}

void BaseSession::step() {
    for(auto &p : priors) p->sample_latents();
    noise->update();
}

std::ostream &BaseSession::printInitStatus(std::ostream &os, std::string indent) {
    os << indent << name << " {\n";
    os << indent << "  Priors: {\n";
    for( auto &p : priors) p->printInitStatus(os, indent + "    ");
    os << indent << "  }\n";
    os << indent << "  Model: {\n";
    model->printInitStatus(os, indent + "    ");
    os << indent << "  }\n";
    os << indent << "  Predictions: {\n";
    pred->printInitStatus(os, indent + "    ");
    os << indent << "  }\n";
    os << indent << "  Noise: ";
    noise->printInitStatus(os, "");
    return os;
}



//--- 

void Session::setSamples(int b, int n) {
  burnin = b;
  nsamples = n;
}

void Session::init() {
  threads_init();
  init_bmrng();
  BaseSession::init();
  if (verbose) {
      printInitStatus(std::cout, "");
      std::cout << "Sampling" << endl;
  }
  iter = 0;
}

void Session::run() {
    init();
    while (iter < burnin + nsamples) {
        step();
        iter++;
    }
}


void Session::step() {
    if (verbose && iter == burnin) {
        printf(" ====== Burn-in complete, averaging samples ====== \n");
    }
    auto starti = tick();
    BaseSession::step();
    auto endi = tick();

    saveModel(iter - burnin);
    printStatus(endi - starti);
}

std::ostream &Session::printInitStatus(std::ostream &os, std::string indent) {
    BaseSession::printInitStatus(os, indent);
    os << indent << "  Samples: " << burnin << " + " << nsamples << "\n";
    if (save_freq > 0) {
        os << indent << "  Save model every: " << save_freq << "\n";
        os << indent << "  Output prefix: " << save_prefix << "\n";
    } else {
        os << indent << "  Don't save model\n";
    }
    os << indent << "}\n";
    return os;
}

volatile bool PythonSession::keepRunning;

void PythonSession::run() {
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

void PythonSession::intHandler(int) {
  keepRunning = false;
  printf("[Received Ctrl-C. Stopping after finishing the current iteration.]\n");
}

// 
//-- cmdline handling stuff
//
template<class SideInfo>
inline void addMacauPrior(Session &m, std::string prior_name, unique_ptr<SideInfo> &features, double lambda_beta, double tol)
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
inline void add_features(MasterPrior<Prior> &p,  std::vector<std::string> fname_features)
{
    for(auto &fname : fname_features) {
        if (fname.find(".sdm") != std::string::npos) {
            auto features = read_sdm(fname.c_str());
            auto &slave_model = p.template addSlave<SparseDenseMF>();
            slave_model.setRelationData(to_eigen(*features));
            delete features;
        } else if (fname.find(".sbm") != std::string::npos) {
            auto features = read_sbm(fname.c_str());
            auto &slave_model = p.template addSlave<SparseDenseMF>();
            slave_model.setRelationData(to_eigen(*features));
            delete features;
        } else if (fname.find(".ddm") != std::string::npos) {
            auto features = read_ddm<Eigen::MatrixXd>(fname.c_str());
            auto &slave_model = p.template addSlave<DenseDenseMF>();
            slave_model.setRelationData(features);
        } else {
            throw std::runtime_error("Train features file: expecing .ddm, .sdm or .sbm, got " + std::string(fname));
        }
    }
}


inline void addMaster(Session &macau,std::string prior_name, std::vector<std::string> fname_features)
{
    if(prior_name == "normal" || prior_name == "default") {
       auto &prior = macau.addPrior<MasterPrior<NormalPrior>>();
       add_features(prior, fname_features);
    } else if(prior_name == "spikeandslab") {
       auto &prior = macau.addPrior<MasterPrior<SpikeAndSlabPrior>>();
       add_features(prior, fname_features);
    } else {
        throw std::runtime_error("Unknown prior with side info: " + prior_name);
    }
}

void add_prior(Session &macau, std::string prior_name, std::vector<std::string> fname_features, double lambda_beta, double tol)
{
    //-- row prior with side information
    if (fname_features.size()) {
        if (prior_name == "macau" || prior_name == "macauone") {
            assert(fname_features.size() == 1);
            auto &fname = fname_features.at(0);
            die_unless_file_exists(fname);
            if (fname.find(".sdm") != std::string::npos) {
                auto row_features = std::unique_ptr<SparseDoubleFeat>(load_csr(fname.c_str()));
                addMacauPrior(macau, prior_name, row_features, lambda_beta, tol);
            } else if (fname.find(".sbm") != std::string::npos) {
                auto features = load_bcsr(fname.c_str());
                addMacauPrior(macau, prior_name, features, lambda_beta, tol);
            } else {
                throw std::runtime_error("Train row_features file: expecing .sdm or .sbm, got " + std::string(fname));
            }
        } else {
            addMaster(macau, prior_name, fname_features);
        }
    } else if(prior_name == "normal" || prior_name == "default") {
        macau.addPrior<NormalPrior>();
    } else if(prior_name == "spikeandslab") {
        macau.addPrior<SpikeAndSlabPrior>();
    } else {
        throw std::runtime_error("Unknown prior without side info: " + prior_name);
    }
}

void Session::setFromConfig(Config &c)
{
    if (c.fname_train.size() == 0) die("Missing --train=FILE");
    die_unless_file_exists(c.fname_train);

    //-- check if fname_test is actually a number
    if ((c.test_split = atof(c.fname_test.c_str())) > .0) {
        c.fname_test.clear();
    }

    // Load main Y matrix file
    if (c.fname_train.find(".sdm") != std::string::npos) {
        auto Ytrain = to_eigen(*read_sdm(c.fname_train.c_str()));
        MF<SparseMatrixD> *model;

        if (is_binary(Ytrain)) {
            model = &sparseBinaryModel(c.num_latent);
            if (!c.classify) {
               c.classify = true;
               c.threshold = 0.5;
            }
        } else {
            model = &sparseModel(c.num_latent);
        }

        if (c.test_split > .0) {
            auto Ytest = extract(Ytrain, c.test_split);
            pred->set(Ytest);
        }
        model->setRelationData(Ytrain);
    } else if (c.fname_train.find(".ddm") != std::string::npos) {
        DenseDenseMF& model = denseDenseModel(c.num_latent);
        auto Ytrain = read_ddm<MatrixXd>(c.fname_train.c_str());
        if (c.test_split > .0) {
            auto Ytest = extract(Ytrain, c.test_split);
            pred->set(Ytest);
        }
        model.setRelationData(Ytrain);
    } else {
        die("Train data file: expecing .sdm or .ddm, got " + std::string(c.fname_train));
    }

    //-- simple options
    setSamples(c.burnin, c.nsamples);
    setVerbose(true);
    setSavePrefix(c.output_prefix);
    setSaveFrequency(c.output_freq);
    if (c.classify) pred->setThreshold(c.threshold);

    //-- noise model
    if(c.adaptive_precision.size() && c.fixed_precision.size()) {
        throw std::runtime_error("Cannot have both --adaptive and --precision");
    }

    if (c.adaptive_precision.size()) {
        char * optarg = strdup(c.adaptive_precision.c_str());
        char *token;
        if(optarg && (token = strsep(&optarg, ","))) c.sn_init = strtod(token, NULL); 
        if(optarg && (token = strsep(&optarg, ","))) c.sn_max = strtod(token, NULL); 
        setAdaptivePrecision(c.sn_init, c.sn_max);
    } else if (c.fixed_precision.size()) {
        c.precision = strtod(c.fixed_precision.c_str(), NULL);
        setPrecision(c.precision);
    } else {
        setPrecision(c.precision);
    }


    // test data
    if (c.fname_test.size()) {
        die_unless_file_exists(c.fname_test);
        if (c.fname_test.find(".sbm") != std::string::npos) {
            auto Ytest = read_sbm(c.fname_test.c_str());
            pred->set(to_eigen(*Ytest));
            delete Ytest;
        } else if (c.fname_test.find(".sdm") != std::string::npos) {
            auto Ytest = read_sdm(c.fname_test.c_str());
            pred->set(*Ytest);
            delete Ytest;
        }
    }

    add_prior(*this, c.col_prior, c.fname_col_features, c.lambda_beta, c.tol);
    add_prior(*this, c.row_prior, c.fname_row_features, c.lambda_beta, c.tol);
}


void Session::printStatus(double elapsedi) {
    if(!verbose) return;

    pred->update(*model, iter < burnin);

    double norm0 = priors[0]->getLinkNorm();
    double norm1 = priors[1]->getLinkNorm();

    double snorm0 = model->U(0).norm();
    double snorm1 = model->U(1).norm();

    auto nnz_per_sec = (model->Ynnz()) / elapsedi;
    auto samples_per_sec = (model->Yrows() + model->Ycols()) / elapsedi;

    std::string phase;
    int i, from;
    if (iter < burnin) {
        phase = "Burnin";
        i = iter;
        from = burnin;
    } else {
        phase = "Sample";
        i = iter - burnin;
        from = nsamples;
    }

    printf("%s %3d/%3d: RMSE: %.4f (1samp: %.4f) AUC:%.4f  U:[%1.2e, %1.2e]  Side:[%1.2e, %1.2e] %s [took %0.1fs, %.0f samples/sec, %.0f nnz/sec]\n",
            phase.c_str(), i, from, pred->rmse_avg, pred->rmse, pred->auc,
            snorm0, snorm1, norm0, norm1, noise->getStatus().c_str(), elapsedi, samples_per_sec, nnz_per_sec);
}

void Session::saveModel(int isample) {
    if (!save_freq || isample < 0) return;
    if ((isample % save_freq) != 0) return;
    string fprefix = save_prefix + "-sample-" + std::to_string(isample);
    model->save(fprefix);
    for(auto &p : priors) p->savePriorInfo(fprefix);
}

} // end namespace Macau
