#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <memory>

#include "mvnormal.h"
#include "linop.h"
#include "model.h"
#include "matrix_io.h"

#include "ILatentPrior.h"

namespace smurff {

// Spike and slab prior
class SpikeAndSlabPrior : public ILatentPrior 
{
public:
   // updated by every thread
   thread_vector<MatrixNNd> Zcol, W2col;

   // read-only during sampling
   MatrixNNd Zkeep;
   ArrayNNd alpha;
   MatrixNNd r;

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
