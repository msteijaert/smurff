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

#include <INIReader.h>

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

void BaseSession::step() {
    for(auto &p : priors) p->sample_latents();
    data->update(model);
}

std::ostream &BaseSession::info(std::ostream &os, std::string indent) {
    os << indent << name << " {\n";
    os << indent << "  Data: {\n";
    data->info(os, indent + "    ");
    os << indent << "  }\n";
    os << indent << "  Model: {\n";
    model.info(os, indent + "    ");
    os << indent << "  }\n";
    os << indent << "  Priors: {\n";
    for( auto &p : priors) p->info(os, indent + "    ");
    os << indent << "  }\n";
    os << indent << "  Result: {\n";
    pred.info(os, indent + "    ");
    os << indent << "  }\n";
    return os;
}

//--- 

void Session::init() {
    threads_init();
    init_bmrng();
    data->init();
    model.init(config.num_latent, data->mean_rating, data->dims());
    for( auto &p : priors) p->init();
    if (config.verbose) info(std::cout, "");
    if (config.restore_prefix.size()) {
        iter = -1;
        if (config.verbose) printf("-- Restoring model, predictions,... from '%s*%s'.\n", config.restore_prefix.c_str(), config.save_suffix.c_str());
        restore(config.restore_prefix, config.restore_suffix);
        if (config.verbose) printStatus(0);
    }
    if (config.verbose) std::cout << "Sampling" << endl;
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
inline void addMacauPrior(Session &m, std::string prior_name, SideInfo *f, double lambda_beta, double tol, int use_FtF)
{
    std::unique_ptr<SideInfo> features(f);

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

void add_prior(Session &macau, std::string prior_name, const std::vector<MatrixConfig> &features, double lambda_beta, double tol, bool direct)
{
    //-- row prior with side information
    if (features.size()) {
        if (prior_name == "macau" || prior_name == "macauone") {
            assert(features.size() == 1);
            auto &s = features.at(0);

            if (s.binary) {
                auto sideinfo = new SparseFeat(s.nrow, s.ncol, s.nnz, s.rows, s.cols);
                addMacauPrior(macau, prior_name, sideinfo, lambda_beta, tol, direct);
            } else if (s.dense) {
                auto sideinfo = new MatrixXd(dense_to_eigen(s));
                addMacauPrior(macau, prior_name, sideinfo, lambda_beta, tol, direct);
            } else {
                auto sideinfo = new SparseDoubleFeat(s.nrow, s.ncol, s.nnz, s.rows, s.cols, s.values);
                addMacauPrior(macau, prior_name, sideinfo, lambda_beta, tol, direct);
            } 
        } else {
            assert(false && "Not yet");
            //addMaster(macau, prior_name, features);
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
    if (!train.rows)              die("Missing train matrix");
    if (test.rows  && test_split) die("Provided both input test pointer and split ratio");

    std::set<std::string> prior_names = { "default", "normal", "spikeandslab", "macau", "macauone" };
    if (prior_names.find(col_prior) == prior_names.end()) die("Unknown col_prior " + col_prior);
    if (prior_names.find(row_prior) == prior_names.end()) die("Unknown row_prior " + row_prior);

    std::set<std::string> noise_models = { "fixed", "adaptive", "probit" };
    if (noise_models.find(noise_model) == noise_models.end()) die("Unknown noise model " + noise_model);

    if (test.nrow > 0 && train.nrow > 0 && test.nrow != train.nrow)
        die("Train and test matrix should have the same number of rows");

    if (test.ncol > 0 && train.ncol > 0 && test.ncol != train.ncol)
        die("Train and test matrix should have the same number of cols");

    std::set<std::string> save_suffixes = { ".csv", ".ddm" };
    if (save_suffixes.find(save_suffix) == save_suffixes.end()) die("Unknown output suffix: " + save_suffix);

    return true;
}

std::ostream &MatrixConfig::info(std::ostream &os) const
{
    os << nrow << " x " << ncol;
    return os;
}

void Config::save(std::string fname) const 
{
    ofstream os(fname);

    os << "# train = "; train.info(os); os << std::endl;
    os << "# test = "; test.info(os); os << std::endl;
    os << "test_split = " << test_split << std::endl;

    os << "# features" << std::endl;
    auto print_features = [&os](std::string name, const std::vector<MatrixConfig> &vec) -> void {
        os << "[" << name << "]\n";
        for (int i=0; i<vec.size(); ++i) {
            os << "# " << i << " ";
            vec.at(i).info(os);
            os << std::endl;
        }
    };
    print_features("row_features", row_features);
    print_features("col_features", col_features);

    os << "# priors" << std::endl;
    os << "row_prior = " << row_prior << std::endl;
    os << "col_prior = " << col_prior << std::endl;

    os << "# restore" << std::endl;
    os << "restore_prefix = " << restore_prefix << std::endl;
    os << "restore_suffix = " << restore_suffix << std::endl;

    os << "# save" << std::endl;
    os << "save_prefix = " << save_prefix << std::endl;
    os << "save_suffix = " << save_suffix << std::endl;
    os << "save_freq = " << save_freq << std::endl;

    os << "# general" << std::endl;
    os << "verbose = " << verbose << std::endl;
    os << "burnin = " << burnin << std::endl;
    os << "nsamples = " << nsamples << std::endl;
    os << "num_latent = " << num_latent << std::endl;

    os << "# for macau priors" << std::endl;
    os << "lambda_beta = " << lambda_beta << std::endl;
    os << "tol = " << tol << std::endl;
    os << "direct = " << direct << std::endl;

    os << "# noise model" << std::endl;
    os << "noise_model = " << noise_model << std::endl;
    os << "precision = " << precision << std::endl;
    os << "sn_init = " << sn_init << std::endl;
    os << "sn_max = " << sn_max << std::endl;

    os << "# binary classification" << std::endl;
    os << "classify = " << classify << std::endl;
    os << "threshold = " << threshold << std::endl;
}

void Config::restore(std::string fname) {
    INIReader reader(fname);
    
    if (reader.ParseError() < 0) {
        std::cout << "Can't load '" << fname << "'\n";
    }

    test_split = reader.GetReal("", "test_split",  .0);

    // -- priors
    row_prior = reader.Get("", "row_prior",  "default");
    col_prior = reader.Get("", "col_prior",  "default");
    
    //-- restore
    restore_prefix = reader.Get("", "restore_prefix",  "");
    restore_suffix = reader.Get("", "restore_suffix",  ".csv");

    //-- save
    save_prefix = reader.Get("", "save_prefix",  "save");
    save_suffix = reader.Get("", "save_suffix",  ".csv");
    save_freq = reader.GetInteger("", "save_freq",  0); // never

    //-- general
    verbose = reader.GetBoolean("", "verbose",  false);
    burnin = reader.GetInteger("", "burnin",  200);
    nsamples = reader.GetInteger("", "nsamples",  800);
    num_latent = reader.GetInteger("", "num_latent",  96);

    //-- for macau priors
    lambda_beta = reader.GetReal("", "lambda_beta",  10.0);
    tol = reader.GetReal("", "tol",  1e-6);
    direct = reader.GetBoolean("", "direct",  false); 

    //-- noise model
    noise_model = reader.Get("", "noise_model",  "fixed");
    precision = reader.GetReal("", "precision",  5.0);
    sn_init = reader.GetReal("", "sn_init",  1.0);
    sn_max = reader.GetReal("", "sn_max",  10.0);

    //-- binary classification
    classify = reader.GetBoolean("", "classify",  false);
    threshold = reader.GetReal("", "threshold",  .0);
};
 

void Session::setFromConfig(const Config &c)
{
    c.validate(true);
    c.save("macau.ini");

    //-- copy
    config = c;

    // Load main Y matrix file
    if (!config.train.dense) {
        SparseMatrixD Ytrain = sparse_to_eigen(config.train);
        if (is_binary(Ytrain)) {
            data = std::unique_ptr<Data>(new ScarceBinaryMatrixData(Ytrain));
            if (!config.classify) {
               config.classify = true;
               config.threshold = 0.5;
               config.noise_model = "probit";
            }
        } else {
            data = std::unique_ptr<Data>(new ScarceMatrixData(Ytrain));
        }

        if (config.test_split > .0) {
            auto predictions = extract(Ytrain, config.test_split);
            pred.set(predictions);
        }
    } else {
        Eigen::MatrixXd Ytrain = dense_to_eigen(config.train);
        data = std::unique_ptr<Data>(new DenseMatrixData(Ytrain));
        if (config.test_split > .0) {
            auto predictions = extract(Ytrain, config.test_split);
            pred.set(predictions);
        }
    }

    if (config.classify) pred.setThreshold(config.threshold);

    //-- noise model
    if (config.noise_model == "adaptive") {
        data->setAdaptivePrecision(config.sn_init, config.sn_max);
    } else if (config.noise_model == "fixed") {
        data->setPrecision(config.precision);
    } else if (config.noise_model == "probit") {
        data->setProbit();
    } else {
        die("Unknown noise model; " + config.noise_model);
    }

    // test data
     pred.set(sparse_to_eigen(config.test));


    add_prior(*this, config.row_prior, config.row_features, config.lambda_beta, config.tol, config.direct);
    add_prior(*this, config.col_prior, config.col_features, config.lambda_beta, config.tol, config.direct);
}


void Session::printStatus(double elapsedi) {
    if(!config.verbose) return;

    pred.update(model, iter < config.burnin);

    double norm0 = priors.at(0)->getLinkNorm();
    double norm1 = priors.at(0)->getLinkNorm();

    double snorm0 = model.U(0).norm();
    double snorm1 = model.U(1).norm();

    // add noise status

    auto nnz_per_sec = (data->nnz()) / elapsedi;
    auto samples_per_sec = (model.nsamples()) / elapsedi;

    std::string phase;
    int i, from;
    if (iter < 0) {
        phase = "Restored state: ";
        i = 0;
        from = 0;
    } else if (iter < config.burnin) {
        phase = "Burnin";
        i = iter;
        from = config.burnin;
    } else {
        phase = "Sample";
        i = iter - config.burnin;
        from = config.nsamples;
    }

    printf("%s %3d/%3d: RMSE: %.4f (1samp: %.4f) AUC:%.4f  U:[%1.2e, %1.2e]  Side:[%1.2e, %1.2e] [took %0.1fs, %.0f samples/sec, %.0f nnz/sec]\n",
            phase.c_str(), i, from, pred.rmse_avg, pred.rmse, pred.auc,
            snorm0, snorm1, norm0, norm1, elapsedi, samples_per_sec, nnz_per_sec);
}

void Session::save(int isample) {
    if (!config.save_freq || isample < 0) return;
    if (((isample+1) % config.save_freq) != 0) return;
    string fprefix = config.save_prefix + "-sample-" + std::to_string(isample);
    if (config.verbose) printf("-- Saving model, predictions,... into '%s*%s'.\n", fprefix.c_str(), config.save_suffix.c_str());
    BaseSession::save(fprefix, config.save_suffix);
}

void BaseSession::save(std::string prefix, std::string suffix) {
    model.save(prefix, suffix);
    pred.save(prefix);
    for(auto &p : priors) p->save(prefix, suffix);
}

void BaseSession::restore(std::string prefix, std::string suffix) {
    model.restore(prefix, suffix);
    for(auto &p : priors) p->restore(prefix, suffix);
}

} // end namespace Macau
