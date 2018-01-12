#pragma once

#include <memory>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <SmurffCpp/Utils/Distribution.h>
#include <SmurffCpp/Utils/ThreadVector.hpp>

#include <SmurffCpp/Priors/ILatentPrior.h>

namespace smurff {

// Spike and slab prior
class SpikeAndSlabPrior : public ILatentPrior 
{
public:
   // updated by every thread
   smurff::thread_vector<Eigen::MatrixXd> Zcol, W2col;

   // read-only during sampling
   Eigen::MatrixXd Zkeep;
   Eigen::ArrayXXd alpha;
   Eigen::MatrixXd r;

   //-- hyper params
   const double prior_beta = 1; //for r
   const double prior_alpha_0 = 1.; //for alpha
   const double prior_beta_0 = 1.; //for alpha

   // hyperparams for CondNormalWishart
   Eigen::VectorXd mu; 
   Eigen::MatrixXd Lambda;
   Eigen::MatrixXd WI;
   Eigen::VectorXd mu0;

   // constants
   int b0;
   int df;

private:
   // for effiency, we keep + update Ucol and UUcol by every thread
   smurff::thread_vector<Eigen::VectorXd> Ucol;
   smurff::thread_vector<Eigen::MatrixXd> UUcol;

private:
  void initUU();

private:
   SpikeAndSlabPrior()
      : ILatentPrior(){}

public:
   SpikeAndSlabPrior(std::shared_ptr<BaseSession> session, uint32_t mode);
   virtual ~SpikeAndSlabPrior() {}
   void init() override;

   void save(std::string prefix, std::string suffix) override;
   void restore(std::string prefix, std::string suffix) override;

   void sample_latent(int n) override;

   void update_prior() override;

   // mean value of Z
   std::ostream &status(std::ostream &os, std::string indent) const override;
};
}
