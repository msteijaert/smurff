#pragma once

#include <memory>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <SmurffCpp/Utils/Distribution.h>

#include <SmurffCpp/Priors/ILatentPrior.h>

namespace smurff {

// Spike and slab prior
class NormalOnePrior : public ILatentPrior 
{
public:
  // hyperparams
  Eigen::VectorXd mu; 
  Eigen::MatrixXd Lambda;
  Eigen::MatrixXd WI;
  Eigen::VectorXd mu0;

  // constants
  int b0;
  int df;

private:
   NormalOnePrior()
      : ILatentPrior(){}

public:
   NormalOnePrior(std::shared_ptr<BaseSession> session, uint32_t mode, std::string name = "NormalOnePrior");
   virtual ~NormalOnePrior() {}
   void init() override;

   void save(std::string prefix, std::string suffix) override {}
   void restore(std::string prefix, std::string suffix) override {}

   void sample_latent(int n) override;
   virtual std::pair<double,double> sample_latent(int d, int k, const Eigen::MatrixXd& XX, const Eigen::VectorXd& yX);

   void update_prior() override;

   // mean value of Z
   std::ostream &status(std::ostream &os, std::string indent) const override;
};
}
