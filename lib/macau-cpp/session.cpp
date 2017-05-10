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

//template<class Prior>
//inline void add_features(MasterPrior<Prior> &p, const std::vector<MatrixConfig> &features)
//{
//    for(auto &f : features) {
//        if (f.dense) {
//            auto &slave_model = p.template addSlave<DenseMatrixData>();
//            slave_model.data = std::unique_ptr<DenseMatrixData>(new DenseMatrixData(dense_to_eigen(f)));
//        } else {
//            auto &slave_model = p.template addSlave<SparseMatrixData>(sparse_to_eigen(f));
//        }
//    }
//}


//inline void addMaster(Session &macau,std::string prior_name, const std::vector<MatrixConfig> &features)
//{
//    if(prior_name == "normal" || prior_name == "default") {
//       auto &prior = macau.addPrior<MasterPrior<NormalPrior>>();
//       add_features(prior, features);
//    } else if(prior_name == "spikeandslab") {
//       auto &prior = macau.addPrior<MasterPrior<SpikeAndSlabPrior>>();
//       add_features(prior, features);
//    } else {
//        throw std::runtime_error("Unknown prior with side info: " + prior_name);
//    }
//}

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

void Session::setFromConfig(const Config &c)
{
    c.validate(true);

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

    double norm0 = priors[0]->getLinkNorm();
    double norm1 = priors[1]->getLinkNorm();

    double snorm0 = model.U(0).norm();
    double snorm1 = model.U(1).norm();

    // add noise status

    auto nnz_per_sec = (data->nnz()) / elapsedi;
    auto samples_per_sec = (model.nsamples()) / elapsedi;

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
