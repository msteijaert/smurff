#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <memory>
#include <iomanip>
#include "utils.h"

#include "model.h"

namespace Macau {

/** interface */
class INoiseModel {
  public:
    INoiseModel(Factors &p) : model(p) {}
    virtual INoiseModel *copyTo(Factors &p) = 0;

    virtual void init()  = 0;
    virtual void update()  = 0;

    virtual std::ostream &printInitStatus(std::ostream &os, std::string indent)   = 0;
    virtual std::string getStatus()  = 0;

    virtual double getAlpha() = 0;

  protected:
    Factors &model;
};

/** Gaussian noise is fixed for the whole run */
class ProbitNoise : public INoiseModel {
  public:
    ProbitNoise(Factors &p) : INoiseModel(p)  {}

    INoiseModel *copyTo(Factors &p) override { return new ProbitNoise(p); }

    void init() override {}
    void update() override {}
    double getAlpha() override { assert(false); }

    std::ostream &printInitStatus(std::ostream &os, std::string indent) override;
    std::string getStatus() override { return std::string(""); }
};

/** Gaussian noise is fixed for the whole run */
class FixedGaussianNoise : public INoiseModel {
  public:
    double alpha;
  
    FixedGaussianNoise(Factors &p, double a = 1.) :
        INoiseModel(p), alpha(a)  {}

    INoiseModel *copyTo(Factors &p) override;

    void init() override { }
    void update() override {}
    double getAlpha() override { return alpha; }

    std::ostream &printInitStatus(std::ostream &os, std::string indent)  override;
    std::string getStatus() override { return std::string(""); }

    void setPrecision(double a) { alpha = a; }    
};

/** Gaussian noise that adapts to the model */
class AdaptiveGaussianNoise : public INoiseModel {
  public:
    double var_total;
    double alpha = NAN;
    double alpha_max = NAN;
    double sn_max;
    double sn_init;

    AdaptiveGaussianNoise(Factors &p, double sinit = 1., double smax = 10.)
        : INoiseModel(p), sn_max(smax), sn_init(sinit) {}

    INoiseModel *copyTo(Factors &) override;

    void init() override;
    void update() override;
    double getAlpha() override { return alpha; }
    void setSNInit(double a) { sn_init = a; }
    void setSNMax(double a) { sn_max  = a; }
    std::ostream &printInitStatus(std::ostream &os, std::string indent) override;
    std::string getStatus() override { return std::string("Prec:") + to_string_with_precision(alpha, 2); }
};

}
