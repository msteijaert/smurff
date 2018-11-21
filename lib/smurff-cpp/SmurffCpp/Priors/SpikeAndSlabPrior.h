#pragma once

#include <memory>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <SmurffCpp/Utils/Distribution.h>
#include <SmurffCpp/Utils/ThreadVector.hpp>

#include <SmurffCpp/Priors/NormalOnePrior.h>

namespace smurff {

// Spike and slab prior
class SpikeAndSlabPrior : public NormalOnePrior 
{
public:
   // updated by every thread during sample_latents
   smurff::thread_vector<Eigen::MatrixXf> Zcol, W2col;

   // updated during update_prior
   Eigen::ArrayXXf Zkeep;
   Eigen::ArrayXXf alpha, log_alpha;
   Eigen::ArrayXXf r, log_r;

   //-- hyper params
   const float prior_beta = 1; //for r
   const float prior_alpha_0 = 1.; //for alpha
   const float prior_beta_0 = 1.; //for alpha

public:
   SpikeAndSlabPrior(std::shared_ptr<Session> session, uint32_t mode);
   virtual ~SpikeAndSlabPrior() {}
   void init() override;

   void restore(std::shared_ptr<const StepFile> sf) override;

   std::pair<float,float> sample_latent(int d, int k, const Eigen::MatrixXf& XX, const Eigen::VectorXf& yX) override;

   void update_prior() override;

   // mean value of Z
   std::ostream &status(std::ostream &os, std::string indent) const override;
};
}
