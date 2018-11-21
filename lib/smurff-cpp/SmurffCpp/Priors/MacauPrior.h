#pragma once

#include <memory>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <SmurffCpp/Priors/NormalPrior.h>

#include <SmurffCpp/SideInfo/ISideInfo.h>

namespace smurff {

//sample_beta method is now virtual. because we override it in MPIMacauPrior
//we also have this method in MacauOnePrior but it is not virtual
//maybe make it virtual?

/// Prior with side information
class MacauPrior : public NormalPrior
{
public:
   std::shared_ptr<Eigen::MatrixXf> 
                   m_beta;            // num_latent x num_feat -- link matrix

   Eigen::MatrixXf Uhat;             // num_latent x num_items
   Eigen::MatrixXf Udelta;           // num_latent x num_items
   Eigen::MatrixXf FtF_plus_precision;    // num_feat   x num feat
   Eigen::MatrixXf HyperU;           // num_latent x num_items
   Eigen::MatrixXf HyperU2;          // num_latent x num_feat
   Eigen::MatrixXf Ft_y;             // num_latent x num_feat -- RHS
   Eigen::MatrixXf BBt;              // num_latent x num_latent

   int blockcg_iter;
   
   float beta_precision_mu0; // Hyper-prior for beta_precision
   float beta_precision_nu0; // Hyper-prior for beta_precision

   std::shared_ptr<ISideInfo> Features;  // side information
   float beta_precision;
   float tol = 1e-6;
   bool use_FtF;
   bool enable_beta_precision_sampling;
   bool throw_on_cholesky_error;

private:
   MacauPrior();

public:
   MacauPrior(std::shared_ptr<Session> session, uint32_t mode);

   virtual ~MacauPrior();

   void init() override;

   void update_prior() override;

   const Eigen::VectorXf getMu(int n) const override;

   Eigen::MatrixXf &beta() const { return *m_beta; }
 
   int num_feat() const { return Features->cols(); }

   void compute_Ft_y(Eigen::MatrixXf& Ft_y);
   virtual void sample_beta();

public:

   void addSideInfo(const std::shared_ptr<ISideInfo>& side_info_a, float beta_precision_a, float tolerance_a, bool direct_a, bool enable_beta_precision_sampling_a, bool throw_on_cholesky_error_a);

public:
   bool save(std::shared_ptr<const StepFile> sf) const override;
   void restore(std::shared_ptr<const StepFile> sf) override;

public:
   std::ostream& info(std::ostream &os, std::string indent) override;
   std::ostream& status(std::ostream &os, std::string indent) const override;

public:
   static std::pair<float, float> posterior_beta_precision(const Eigen::MatrixXf & BBt, Eigen::MatrixXf & Lambda_u, float nu, float mu, int N);
   static float sample_beta_precision(const Eigen::MatrixXf & BBt, Eigen::MatrixXf & Lambda_u, float nu, float mu, int N);
};

}
