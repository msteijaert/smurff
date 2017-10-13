#pragma once

#include <memory>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <Utils/Distribution.h>

#include "ILatentPrior.h"

#include "model.h"

namespace smurff {

// Spike and slab prior
class SpikeAndSlabPrior : public ILatentPrior 
{
public:
   // updated by every thread
   thread_vector<Eigen::MatrixXd> Zcol, W2col;

   // read-only during sampling
   Eigen::MatrixXd Zkeep;
   Eigen::ArrayXXd alpha;
   Eigen::MatrixXd r;

   //-- hyper params
   const double prior_beta = 1; //for r
   const double prior_alpha_0 = 1.; //for alpha
   const double prior_beta_0 = 1.; //for alpha

public:
   SpikeAndSlabPrior(BaseSession &m, int p);
   virtual ~SpikeAndSlabPrior() {}
   void init() override;

   void save(std::string prefix, std::string suffix) override {}
   void restore(std::string prefix, std::string suffix) override {}

   void sample_latents() override;
   void sample_latent(int n) override;

   // mean value of Z
   std::ostream &status(std::ostream &os, std::string indent) const override;
};
}
