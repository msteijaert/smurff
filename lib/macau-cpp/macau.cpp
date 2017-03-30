#include <Eigen/Sparse>

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
    os << indent << "  Factors: {\n";
    model->printInitStatus(os, indent + "    ");
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
  if (save_freq) {
      model->saveGlobalParams(save_prefix);
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
    os << indent << "  Save model every: " << save_freq << "\n";
    os << indent << "  Output prefix: " << save_prefix << "\n";
    os << indent << "  Threshold for binary classification : " << threshold << "\n";
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

void Session::setFromArgs(int argc, char** argv, bool print) {
    std::string fname_train;
    std::string fname_test;
    std::vector<std::string> fname_row_features;
    std::vector<std::string> fname_col_features;
    std::string row_prior("default");
    std::string col_prior("default");
    std::string output_prefix;
    int output_freq = 0;

    double precision          = 5.0;
    bool fixed_precision      = false;

    char *token;
    double sn_init            = 1.0;
    double sn_max             = 10.0;
    bool adaptive_precision   = false;

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
        "  --adaptive 1.0,10.0  adavtive precision of observations\n"
        "  --lambda-beta  10.0  initial value of lambda beta\n"
        "  --tol          1e-6  tolerance for CG\n"
        "  --output-prefix prx  prefix for result files\n"
        "  --output-freq     0  save every n iterations (0 == never)\n\n";

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
            {"adaptive",     required_argument, 0, 'v'},
            {"burnin",       required_argument, 0, 'b'},
            {"nsamples",     required_argument, 0, 'n'},
            {"output-prefix",required_argument, 0, 'o'},
            {"output-freq",  required_argument, 0, 'm'},
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
            case 'm': output_freq        = strtol(optarg, NULL, 10); break;
            case 'p': 
                      if(adaptive_precision) throw std::runtime_error("Cannot have both --adaptive and --precision");
                      precision          = strtod(optarg, NULL);
                      fixed_precision    = true;
                      break;
            case 'v': 
                      if(fixed_precision) throw std::runtime_error("Cannot have both --adaptive and --precision");
                      if(optarg && (token = strsep(&optarg, ","))) sn_init = strtod(token, NULL); 
                      if(optarg && (token = strsep(&optarg, ","))) sn_max = strtod(token, NULL); 
                      adaptive_precision = true;
                      break;
            case 'r': fname_row_features.push_back(optarg); break;
            case 'f': fname_col_features.push_back(optarg); break;
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

    //-- check if fname_test is actually a number
    if ((test_split = atof(fname_test.c_str())) > .0) {
        fname_test.clear();
    }

    // Load main Y matrix file
    if (fname_train.find(".sbm") != std::string::npos) {
        SparseBinaryMF& model = sparseBinaryModel(num_latent);
        auto Ytrain = to_eigen(*read_sbm(fname_train.c_str()));
        if (test_split > .0) {
            auto Ytest = extract(Ytrain, test_split);
            model.setRelationDataTest(Ytest);
        }
        model.setRelationData(Ytrain);
        setThreshold(0.5);
    } else if (fname_train.find(".sdm") != std::string::npos) {
        SparseMF& model = sparseModel(num_latent);
        auto Ytrain = to_eigen(*read_sdm(fname_train.c_str()));
        if (test_split > .0) {
            auto Ytest = extract(Ytrain, test_split);
            model.setRelationDataTest(Ytest);
        }
        model.setRelationData(Ytrain);
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

    if (adaptive_precision) setAdaptivePrecision(sn_init, sn_max);
    else setPrecision(precision);

    setVerbose(true);

    // test data
    if (fname_test.size()) {
        die_unless_file_exists(fname_test);
        if (fname_test.find(".sbm") != std::string::npos) {
            auto Ytest = read_sbm(fname_test.c_str());
            model->setRelationDataTest(to_eigen(*Ytest));
            delete Ytest;
        } else if (fname_test.find(".sdm") != std::string::npos) {
            auto Ytest = read_sdm(fname_test.c_str());
            model->setRelationDataTest(*Ytest);
            delete Ytest;
        }
    }

    add_prior(*this, col_prior, fname_col_features, lambda_beta, tol);
    add_prior(*this, row_prior, fname_row_features, lambda_beta, tol);

    setSavePrefix(output_prefix);
    setSaveFrequency(output_freq);
}


void Session::printStatus(double elapsedi) {
    if(!verbose) return;
    double norm0 = priors[0]->getLinkNorm();
    double norm1 = priors[1]->getLinkNorm();

    double snorm0 = model->U(0).norm();
    double snorm1 = model->U(1).norm();

    std::pair<double,double> rmse_test = model->getRMSE(iter, burnin);

    double auc = model->auc(threshold);

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
            phase.c_str(), i, from, 
            rmse_test.second, rmse_test.first, auc,
            snorm0, snorm1, norm0, norm1, noise->getStatus().c_str(), elapsedi, samples_per_sec, nnz_per_sec);
}

void Session::saveModel(int isample) {
    if (!save_freq || isample < 0) return;
    if ((isample % save_freq) != 0) return;
    string fprefix = save_prefix + "-sample-" + std::to_string(isample);
    model->saveModel(fprefix, isample, burnin);
    for(auto &p : priors) p->savePriorInfo(fprefix);
}

} // end namespace Macau
