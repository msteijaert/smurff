#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <memory>
#include <iomanip>
#include "utils.h"

namespace Macau {

struct Data;
struct Model;

/** interface */
class INoiseModel {
  public:
    INoiseModel(Data &p) : data(p) {}
    virtual void init()  = 0;
    virtual void update(const Model &)  = 0;

    virtual std::ostream &info(std::ostream &os, std::string indent)   = 0;
    virtual std::string getStatus()  = 0;

    virtual double getAlpha() = 0;

  protected:
    Data &data;
};

/** Gaussian noise is fixed for the whole run */
class FixedGaussianNoise : public INoiseModel {
  public:
    double alpha;
  
    FixedGaussianNoise(Data &p, double a = 1.) :
        INoiseModel(p), alpha(a)  {}

    void init() override { }
    void update(const Model &) override {}
    double getAlpha() override { return alpha; }

    std::ostream &info(std::ostream &os, std::string indent)  override;
    std::string getStatus() override { return std::string(""); }

    void setPrecision(double a) { alpha = a; }    
};

/** Gaussian noise that adapts to the data */
class AdaptiveGaussianNoise : public INoiseModel {
  public:
    double var_total;
    double alpha = NAN;
    double alpha_max = NAN;
    double sn_max;
    double sn_init;

    AdaptiveGaussianNoise(Data &p, double sinit = 1., double smax = 10.)
        : INoiseModel(p), sn_max(smax), sn_init(sinit) {}

    void init() override;
    void update(const Model &) override;
    double getAlpha() override { return alpha; }
    void setSNInit(double a) { sn_init = a; }
    void setSNMax(double a) { sn_max  = a; }
    std::ostream &info(std::ostream &os, std::string indent) override;
    std::string getStatus() override { return std::string("Prec:") + to_string_with_precision(alpha, 2); }
};

/** Gaussian noise that adapts to the data */
class ProbitNoise : public INoiseModel {
  public:
    ProbitNoise(Data &p) : INoiseModel(p) {}

    void init() override {}
    void update(const Model &) override {}
    double getAlpha() override { assert(false); return NAN; }
    std::ostream &info(std::ostream &os, std::string indent) override;
    std::string getStatus() override { return std::string(); }
};

}
