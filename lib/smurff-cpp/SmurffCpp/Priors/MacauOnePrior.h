#pragma once

#include <memory>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <SmurffCpp/Priors/NormalOnePrior.h>

#include <SmurffCpp/SideInfo/ISideInfo.h>

namespace smurff {

//Why remove init method and put everything in constructor if we have
//init method in other priors and the other method addSideInfo which we use in pair

class MacauOnePrior : public NormalOnePrior
{
public:
   Eigen::MatrixXf Uhat;

   Eigen::VectorXf F_colsq;   // sum-of-squares for every feature (column)

   Eigen::MatrixXf beta;      // link matrix
   
   float beta_precision_a0; // Hyper-prior for beta_precision
   float beta_precision_b0; // Hyper-prior for beta_precision

   //FIXME: these must be used

   //new values

   std::vector<std::shared_ptr<ISideInfo> > side_info_values;
   std::vector<float> beta_precision_values;
   std::vector<bool> enable_beta_precision_sampling_values;

   //FIXME: these must be removed

   //old values

   std::shared_ptr<ISideInfo> Features;  // side information
   Eigen::VectorXf beta_precision;
   float bp0;
   bool enable_beta_precision_sampling;

public:
   MacauOnePrior(std::shared_ptr<Session> session, uint32_t mode);

   void init() override;

   void update_prior() override;
    
   const Eigen::VectorXf getMu(int n) const override;

public:
   //FIXME: tolerance_a and direct_a are not really used. 
   //should remove later after PriorFactory is properly implemented. 
   //No reason generalizing addSideInfo between priors
   void addSideInfo(const std::shared_ptr<ISideInfo>& side_info_a, float beta_precision_a, float tolerance_a, bool direct_a, bool enable_beta_precision_sampling_a, bool throw_on_cholesky_error_a);

public:

   //used in update_prior

   void sample_beta(const Eigen::MatrixXf &U);

   //used in update_prior

   void sample_mu_lambda(const Eigen::MatrixXf &U);

   //used in update_prior

   void sample_beta_precision();

public:

   bool save(std::shared_ptr<const StepFile> sf) const override;

   void restore(std::shared_ptr<const StepFile> sf) override;

   std::ostream& status(std::ostream &os, std::string indent) const override;
};

}
