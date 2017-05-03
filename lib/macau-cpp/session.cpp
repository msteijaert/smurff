#include <set>
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

std::ostream &BaseSession::info(std::ostream &os, std::string indent) {
    os << indent << name << " {\n";
    os << indent << "  Priors: {\n";
    for( auto &p : priors) p->info(os, indent + "    ");
    os << indent << "  }\n";
    os << indent << "  Model: {\n";
    model->info(os, indent + "    ");
    os << indent << "  }\n";
    os << indent << "  Result: {\n";
    pred.info(os, indent + "    ");
    os << indent << "  }\n";
    os << indent << "  Noise: ";
    noise->info(os, "");
    return os;
}



//--- 

void Session::init() {
    threads_init();
    init_bmrng();
    BaseSession::init();
    if (config.restore_prefix.size()) {
        if (config.verbose) printf("-- Restoring model, predictions,... from '%s*%s'.\n", config.restore_prefix.c_str(), config.save_suffix.c_str());
        restore(config.restore_prefix, config.restore_suffix);
    }
    if (config.verbose) {
        info(std::cout, "");
        std::cout << "Sampling" << endl;
    }
    iter = 0;
    is_init = true;
}

void Session::run() {
    init();
    while (iter < config.burnin + config.nsamples) step();
}


void Session::step() {
    assert(is_init);
    if (config.verbose && iter == config.burnin) {
        printf(" ====== Burn-in complete, averaging samples ====== \n");
    }
    auto starti = tick();
    BaseSession::step();
    auto endi = tick();

    printStatus(endi - starti);
    save(iter - config.burnin);
    iter++;
}

std::ostream &Session::info(std::ostream &os, std::string indent) {
    BaseSession::info(os, indent);
    os << indent << "  Iterations: " << config.burnin << " burnin + " << config.nsamples << " samples\n";
    if (config.save_freq > 0) {
        os << indent << "  Save model: every " << config.save_freq << " iteration\n";
        os << indent << "  Save prefix: " << config.save_prefix << "\n";
        os << indent << "  Save suffix: " << config.save_suffix << "\n";
    } else {
        os << indent << "  Save model: never\n";
    }
    if (config.restore_prefix.size()) {
        os << indent << "  Restore prefix: " << config.restore_prefix << "\n";
        os << indent << "  Restore suffix: " << config.restore_suffix << "\n";
    }
    os << indent << "}\n";
    return os;
}

bool PythonSession::keepRunning = true;

void PythonSession::step() {
    if (!keepRunning) return;
    signal(SIGINT, intHandler);
    Session::step();
}

void PythonSession::intHandler(int) {
  keepRunning = false;
  printf("[Received Ctrl-C. Stopping after finishing the current iteration.]\n");
}

// 
//-- cmdline handling stuff
//
template<class SideInfo>
inline void addMacauPrior(Session &m, std::string prior_name, unique_ptr<SideInfo> &features, double lambda_beta, double tol, int use_FtF)
{
    if(prior_name == "macau" || prior_name == "default"){
        auto &prior = m.addPrior<MacauPrior<SideInfo>>();
        prior.addSideInfo(features, use_FtF);
        prior.setLambdaBeta(lambda_beta);
        prior.setTol(tol);
    } else if(prior_name == "macauone") {
        auto &prior = m.addPrior<MacauOnePrior<SideInfo>>();
        prior.addSideInfo(features, use_FtF);
        prior.setLambdaBeta(lambda_beta);
    } else {
        throw std::runtime_error("Unknown prior with side info: " + prior_name);
    }
}

template<class Prior>
inline void add_features(MasterPrior<Prior> &p,  std::vector<std::string> fname_features)
{
    for(auto &fname : fname_features) {
        assert(is_matrix_fname(fname));
        if (is_sparse_fname(fname)) {
            auto &slave_model = p.template addSlave<SparseDenseMF>();
            read_sparse(fname, slave_model.Y);
        } else {
            auto &slave_model = p.template addSlave<DenseDenseMF>();
            read_dense(fname, slave_model.Y);
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

void add_prior(Session &macau, std::string prior_name, std::vector<std::string> fname_features, double lambda_beta, double tol, bool direct)
{
    //-- row prior with side information
    if (fname_features.size()) {
        if (prior_name == "macau" || prior_name == "macauone") {
            assert(fname_features.size() == 1);
            auto &fname = fname_features.at(0);
            if (fname.find(".sdm") != std::string::npos) {
                auto features = load_csr(fname.c_str());
                addMacauPrior(macau, prior_name, features, lambda_beta, tol, direct);
            } else if (fname.find(".sbm") != std::string::npos) {
                auto features = load_bcsr(fname.c_str());
                addMacauPrior(macau, prior_name, features, lambda_beta, tol, direct);
            } else if (is_dense_fname(fname)) {
                auto features = std::unique_ptr<MatrixXd>(new MatrixXd);
                read_dense(fname.c_str(), *features);
                addMacauPrior(macau, prior_name, features, lambda_beta, tol, true);
            } else {
                throw std::runtime_error("Train features file: expecing .sdm or .sbm, got " + std::string(fname));
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

bool Config::validate(bool throw_error) const 
{
    if (!fname_train.size() && !config_train.rows) die("Missing train matrix");
    if (fname_train.size() && config_train.rows) die("Provided both input train pointer and input train file");

    if (fname_test.size() && config_test.rows) die("Provided both input test pointer and input test file");
    if (fname_test.size() && test_split)       die("Provided both input test file and split ratio");
    if (config_test.rows  && test_split)       die("Provided both input test pointer and split ratio");

    if (config_row_features.size() && fname_row_features.size()) die("Provided both row-features file and pointer");
    if (config_col_features.size() && fname_col_features.size()) die("Provided both col-features file and pointer");

    std::set<std::string> prior_names = { "default", "normal", "spikeandslab", "macau", "macauone" };
    if (prior_names.find(col_prior) == prior_names.end()) die("Unknown col_prior " + col_prior);
    if (prior_names.find(row_prior) == prior_names.end()) die("Unknown row_prior " + row_prior);

    std::set<std::string> noise_models = { "fixed", "adaptive", "probit" };
    if (noise_models.find(noise_model) == noise_models.end()) die("Unknown noise model " + noise_model);

    if (config_test.rows > 0 && config_train.rows > 0 && config_test.rows != config_train.rows)
        die("Train and test matrix should have the same number of rows");

    if (config_test.cols > 0 && config_train.cols > 0 && config_test.cols != config_train.cols)
        die("Train and test matrix should have the same number of cols");

    std::set<std::string> save_suffixes = { ".csv", ".ddm" };
    if (save_suffixes.find(save_suffix) == save_suffixes.end()) die("Unknown output suffix: " + save_suffix);

    return true;
 }

void Session::setFromConfig(const Config &c)
{
    c.validate(true);

    //-- copy
    config = c;

    bool train_is_sparse = is_sparse_fname(config.fname_train) || (!config.config_train.dense);

    // Load main Y matrix file
    if (train_is_sparse) {
        SparseMatrixD Ytrain;
        if (config.fname_train.size()) {
            read_sparse(config.fname_train, Ytrain);
        } else {
            Ytrain = to_eigen(config.config_train);
        }

        MF<SparseMatrixD> *model;

        if (is_binary(Ytrain)) {
            model = &sparseBinaryModel(config.num_latent);
            if (!config.classify) {
               config.classify = true;
               config.threshold = 0.5;
            }
        } else {
            model = &sparseModel(config.num_latent);
        }

        if (config.test_split > .0) {
            auto predictions = extract(Ytrain, config.test_split);
            pred.set(predictions);
        }
        model->setRelationData(Ytrain);
    } else {
        DenseDenseMF& model = denseDenseModel(config.num_latent);
        auto Ytrain = model.Y;
        read_dense(config.fname_train, Ytrain);
        if (config.test_split > .0) {
            auto predictions = extract(Ytrain, config.test_split);
            pred.set(predictions);
        }
        model.setRelationData(Ytrain);
    }

    if (config.classify) pred.setThreshold(config.threshold);

    //-- noise model
    if (config.noise_model == "adaptive") {
        setAdaptivePrecision(config.sn_init, config.sn_max);
    } else if (config.noise_model == "fixed") {
        setPrecision(config.precision);
    } else {
        die("Unknown noise model; " + config.noise_model);
    }

    // test data
    if (config.fname_test.size()) {
        die_unless_file_exists(config.fname_test);
        if (config.fname_test.find(".sbm") != std::string::npos) {
            auto predictions = read_sbm(config.fname_test.c_str());
            pred.set(to_eigen(*predictions));
            delete predictions;
        } else if (config.fname_test.find(".sdm") != std::string::npos) {
            auto predictions = read_sdm(config.fname_test.c_str());
            pred.set(*predictions);
            delete predictions;
        }
    } else if (config.config_test.nrow > 0) {
         pred.set(to_eigen(config.config_train));
    }

    add_prior(*this, config.col_prior, config.fname_col_features, config.lambda_beta, config.tol, config.direct);
    add_prior(*this, config.row_prior, config.fname_row_features, config.lambda_beta, config.tol, config.direct);
}


void Session::printStatus(double elapsedi) {
    if(!config.verbose) return;

    pred.update(*model, iter < config.burnin);

    double norm0 = priors[0]->getLinkNorm();
    double norm1 = priors[1]->getLinkNorm();

    double snorm0 = model->U(0).norm();
    double snorm1 = model->U(1).norm();

    auto nnz_per_sec = (model->Ynnz()) / elapsedi;
    auto samples_per_sec = (model->Yrows() + model->Ycols()) / elapsedi;

    std::string phase;
    int i, from;
    if (iter < config.burnin) {
        phase = "Burnin";
        i = iter;
        from = config.burnin;
    } else {
        phase = "Sample";
        i = iter - config.burnin;
        from = config.nsamples;
    }

    printf("%s %3d/%3d: RMSE: %.4f (1samp: %.4f) AUC:%.4f  U:[%1.2e, %1.2e]  Side:[%1.2e, %1.2e] %s [took %0.1fs, %.0f samples/sec, %.0f nnz/sec]\n",
            phase.c_str(), i, from, pred.rmse_avg, pred.rmse, pred.auc,
            snorm0, snorm1, norm0, norm1, noise->getStatus().c_str(), elapsedi, samples_per_sec, nnz_per_sec);
}

void Session::save(int isample) {
    if (!config.save_freq || isample < 0) return;
    if (((isample+1) % config.save_freq) != 0) return;
    string fprefix = config.save_prefix + "-sample-" + std::to_string(isample);
    if (config.verbose) printf("-- Saving model, predictions,... into '%s*%s'.\n", fprefix.c_str(), config.save_suffix.c_str());
    BaseSession::save(fprefix, config.save_suffix);
}

void BaseSession::save(std::string prefix, std::string suffix) {
    model->save(prefix, suffix);
    pred.save(prefix);
    for(auto &p : priors) p->save(prefix, suffix);
}

void BaseSession::restore(std::string prefix, std::string suffix) {
    model->restore(prefix, suffix);
    for(auto &p : priors) p->restore(prefix, suffix);
}

} // end namespace Macau
