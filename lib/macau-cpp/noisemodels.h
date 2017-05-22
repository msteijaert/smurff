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
    INoiseModel(Model &p) : model(p) {}
    virtual INoiseModel *copyTo(Model &p) = 0;

    virtual void init()  = 0;
    virtual void update()  = 0;

    virtual std::ostream &info(std::ostream &os, std::string indent)   = 0;
    virtual std::string getStatus()  = 0;

    virtual double getAlpha() = 0;

  protected:
    Model &model;
};

/** Gaussian noise is fixed for the whole run */
class FixedGaussianNoise : public INoiseModel {
  public:
    double alpha;
  
    FixedGaussianNoise(Model &p, double a = 1.) :
        INoiseModel(p), alpha(a)  {}

    INoiseModel *copyTo(Model &p) override;

    void init() override { }
    void update() override {}
    double getAlpha() override { return alpha; }

    std::ostream &info(std::ostream &os, std::string indent)  override;
    std::string getStatus() override { return std::string("Fixed: ") = std::to_string(alpha); }

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

    AdaptiveGaussianNoise(Model &p, double sinit = 1., double smax = 10.)
        : INoiseModel(p), sn_max(smax), sn_init(sinit) {}

    INoiseModel *copyTo(Model &) override;

    void init() override;
    void update() override;
    double getAlpha() override { return alpha; }
    void setSNInit(double a) { sn_init = a; }
    void setSNMax(double a) { sn_max  = a; }
    std::ostream &info(std::ostream &os, std::string indent) override;
    std::string getStatus() override { return std::string("Prec:") + to_string_with_precision(alpha, 2); }
};

}
