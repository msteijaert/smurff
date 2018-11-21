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
  Eigen::VectorXf mu; 
  Eigen::MatrixXf Lambda;
  Eigen::MatrixXf WI;
  Eigen::VectorXf mu0;

  // constants
  int b0;
  int df;

private:
   NormalOnePrior()
      : ILatentPrior(){}

public:
   NormalOnePrior(std::shared_ptr<Session> session, uint32_t mode, std::string name = "NormalOnePrior");
   virtual ~NormalOnePrior() {}
   void init() override;

   //mu in NormalPrior does not depend on column index
   //however successors of this class can override this method
   //for example in MacauPrior mu depends on Uhat.col(n)
   virtual const Eigen::VectorXf getMu(int n) const;

   void sample_latent(int n) override;
   virtual std::pair<float,float> sample_latent(int d, int k, const Eigen::MatrixXf& XX, const Eigen::VectorXf& yX);

   void update_prior() override;

   // mean value of Z
   std::ostream &status(std::ostream &os, std::string indent) const override;
};
}
