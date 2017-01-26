#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <memory>
#include <iomanip>
#include "utils.h"

#include "model.h"

/** interface */
class INoiseModel {
  public:
    INoiseModel(Factors &p) : model(p) {}

    virtual void init()  = 0;
    virtual void update()  = 0;

    virtual std::string getInitStatus()   = 0;
    virtual std::string getStatus()  = 0;

    virtual double getAlpha() = 0;

  protected:
    Factors &model;
};

/** Gaussian noise is fixed for the whole run */
class FixedGaussianNoise : public INoiseModel {
  public:
    double alpha;
    double rmse_test;
    double rmse_test_onesample;
  
    FixedGaussianNoise(Factors &p, double a = 1.) :
        INoiseModel(p), alpha(a)  {}

    void init() override { }
    void update() override {}
    double getAlpha() override { return alpha; }

    std::string getInitStatus()  override { return std::string("Noise precision: ") + std::to_string(alpha) + " (fixed)"; }
    std::string getStatus() override { return std::string(""); }

    void setPrecision(double a) { alpha = a; }    
};

/** Gaussian noise that adapts to the model */
class AdaptiveGaussianNoise : public INoiseModel {
  public:
    double alpha = NAN;
    double alpha_max = NAN;
    double sn_max;
    double sn_init;
    double var_total = NAN;

    AdaptiveGaussianNoise(Factors &p, double sinit = 1., double smax = 10.)
        : INoiseModel(p), sn_max(smax), sn_init(sinit) {}

    void init() override;
    void update() override;
    double getAlpha() override { return alpha; }

    void setSNInit(double a) { sn_init = a; }
    void setSNMax(double a) { sn_max  = a; }
    std::string getInitStatus() override { return "Noise precision: adaptive (with max precision of " + std::to_string(alpha_max) + ")"; }
    std::string getStatus() override { return std::string("Prec:") + to_string_with_precision(alpha, 2); }
};

