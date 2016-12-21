#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <memory>
#include <iomanip>

#include "latentprior.h"

class ILatentPrior; // forward declaration

/** interface */
class INoiseModel {
  public:
    INoiseModel(const ILatentPrior &_U, ILatentPrior &_V) : U(_U), V(_V) {}
    const ILatentPrior &U, &V;
};

/** Gaussian noise is fixed for the whole run */
class FixedGaussianNoise : public INoiseModel {
  public:
    double alpha;
    double rmse_test;
    double rmse_test_onesample;
  
    FixedGaussianNoise(const ILatentPrior &_U, ILatentPrior &_V, double a = 1.) :
        INoiseModel(_U, _V), alpha(a)  {}

    void init() { }
    void update() {}
    std::pair<double, double> sample(int, int) { return std::make_pair(alpha, alpha); }

    std::string getInitStatus() { return std::string("Noise precision: ") + std::to_string(alpha) + " (fixed)"; }
    std::string getStatus() { return std::string(""); }

    void setPrecision(double a) { alpha = a; }    
    void evalModel(Eigen::SparseMatrix<double> & Ytest, const int n, Eigen::VectorXd & predictions, Eigen::VectorXd & predictions_var);
    double getEvalMetric() { return rmse_test;}
    std::string getEvalString() { return std::string("RMSE: ") + to_string_with_precision(rmse_test,5) + " (1samp: " + to_string_with_precision(rmse_test_onesample,5)+")";}
 
};

/** Gaussian noise that adapts to the data */
class AdaptiveGaussianNoise : public INoiseModel {
  public:
    double alpha = NAN;
    double alpha_max = NAN;
    double sn_max;
    double sn_init;
    double var_total = NAN;
    double rmse_test;
    double rmse_test_onesample;

    AdaptiveGaussianNoise(const ILatentPrior &_U, ILatentPrior &_V, double sinit = 1., double smax = 10.)
        : INoiseModel(_U, _V), sn_max(smax), sn_init(sinit) {}

    void init();
    void update();
    std::pair<double,double> sample(int n, int m) { return std::make_pair(alpha, alpha); }

    void setSNInit(double a) { sn_init = a; }
    void setSNMax(double a)  { sn_max  = a; }
    std::string getInitStatus() { return std::string("Noise precision: adaptive (with max precision of ") + std::to_string(alpha_max) + ")"; }

    std::string getStatus() {
      return std::string("Prec:") + to_string_with_precision(alpha, 2);
    }

    void evalModel(Eigen::SparseMatrix<double> & Ytest, const int n, Eigen::VectorXd & predictions, Eigen::VectorXd & predictions_var);
    double getEvalMetric() {return rmse_test;}
    std::string getEvalString() { return std::string("RMSE: ") + to_string_with_precision(rmse_test,5) + " (1samp: " + to_string_with_precision(rmse_test_onesample,5)+")";}
};

/** Probit noise model (binary). Fixed for the whole run */
class ProbitNoise : public INoiseModel {
  public:
    double auc_test;
    double auc_test_onesample;
    ProbitNoise(const ILatentPrior &_U, ILatentPrior &_V)
        : INoiseModel(_U, _V) {}

    void init() { }
    void update() {}

    std::pair<double, double> sample(int, int);
        
    std::string getInitStatus() { return std::string("Probit noise model"); }
    std::string getStatus() { return std::string(""); }
    void evalModel(Eigen::SparseMatrix<double> & Ytest, const int n, Eigen::VectorXd & predictions, Eigen::VectorXd & predictions_var);
    double getEvalMetric() {return auc_test;}
    std::string getEvalString() { return std::string("AUC: ") + to_string_with_precision(auc_test,5) + " (1samp: " + to_string_with_precision(auc_test_onesample,5)+")";}
};
