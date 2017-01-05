#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <memory>
#include <iomanip>
#include "bpmfutils.h"

struct MFactors; // forward declaration

/** interface */
class INoiseModel {
  public:
    INoiseModel(MFactors &p) : model(p) {}

    virtual void init()  = 0;
    virtual void update()  = 0;

    virtual std::string getInitStatus()   = 0;
    virtual std::string getStatus()  = 0;
    virtual void evalModel(bool burnin)  = 0;
    virtual double getEvalMetric()  = 0;
    virtual std::string getEvalString()  = 0;

    virtual std::pair<double, double> sample(int, int) = 0;

  protected:
    MFactors &model;
};

/** Gaussian noise is fixed for the whole run */
class FixedGaussianNoise : public INoiseModel {
  public:
    double alpha;
    double rmse_test;
    double rmse_test_onesample;
  
    FixedGaussianNoise(MFactors &p, double a = 1.) :
        INoiseModel(p), alpha(a)  {}

    void init() override { }
    void update() override {}
    std::pair<double, double> sample(int, int) override { return std::make_pair(alpha, alpha); }

    std::string getInitStatus()  override { return std::string("Noise precision: ") + std::to_string(alpha) + " (fixed)"; }
    std::string getStatus() override { return std::string(""); }

    void setPrecision(double a) { alpha = a; }    
    void evalModel(bool burnin) override;
    double getEvalMetric() override { return rmse_test;}
    std::string getEvalString() override { return std::string("RMSE: ") + to_string_with_precision(rmse_test,5) + " (1samp: " + to_string_with_precision(rmse_test_onesample,5)+")";}
 
};

/** Gaussian noise that adapts to the model */
class AdaptiveGaussianNoise : public INoiseModel {
  public:
    double alpha = NAN;
    double alpha_max = NAN;
    double sn_max;
    double sn_init;
    double var_total = NAN;
    double rmse_test;
    double rmse_test_onesample;

    AdaptiveGaussianNoise(MFactors &p, double sinit = 1., double smax = 10.)
        : INoiseModel(p), sn_max(smax), sn_init(sinit) {}

    void init() override;
    void update() override;
    std::pair<double,double> sample(int n, int m) override { return std::make_pair(alpha, alpha); }

    void setSNInit(double a) { sn_init = a; }
    void setSNMax(double a) { sn_max  = a; }
    std::string getInitStatus() override { return "Noise precision: adaptive (with max precision of " + std::to_string(alpha_max) + ")"; }
    std::string getStatus() override { return std::string("Prec:") + to_string_with_precision(alpha, 2); }

    void evalModel(bool burnin) override;
    double getEvalMetric() override {return rmse_test;}
    std::string getEvalString() override { return std::string("RMSE: ") + to_string_with_precision(rmse_test,5) + " (1samp: " + to_string_with_precision(rmse_test_onesample,5)+")";}
};

/** Probit noise model (binary). Fixed for the whole run */
class ProbitNoise : public INoiseModel {
  public:
    double auc_test;
    double auc_test_onesample;
    ProbitNoise(MFactors &p) : INoiseModel(p) {}
    void init() override {}
    void update() override {}

    std::pair<double, double> sample(int, int) override;
        
    std::string getInitStatus() override { return std::string("Probit noise model"); }
    std::string getStatus() override { return std::string(""); }
    void evalModel(bool burnin) override;
    double getEvalMetric() override {return auc_test;}
    std::string getEvalString() override { return std::string("AUC: ") + to_string_with_precision(auc_test,5) + " (1samp: " + to_string_with_precision(auc_test_onesample,5)+")";}
};
