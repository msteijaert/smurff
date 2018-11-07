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
   std::shared_ptr<Eigen::MatrixXd> 
                   m_beta;            // num_feat x num_latent -- link matrix

   Eigen::MatrixXd Uhat;              // num_latent x num_items
   Eigen::MatrixXd Udelta;            // num_latent x num_items
   Eigen::MatrixXd FtF_plus_precision;// num_feat   x num feat
   Eigen::MatrixXd HyperU;            // num_latent x num_items
   Eigen::MatrixXd HyperU2;           // num_latent x num_feat
   Eigen::MatrixXd Ft_y;              // num_latent x num_feat -- RHS
   Eigen::MatrixXd BBt;               // num_latent x num_latent

   int blockcg_iter;
   
   double beta_precision_mu0; // Hyper-prior for beta_precision
   double beta_precision_nu0; // Hyper-prior for beta_precision

   //FIXME: these must be used

   //new values

   std::vector<std::shared_ptr<ISideInfo> > side_info_values;
   std::vector<double> beta_precision_values;
   std::vector<double> tol_values;
   std::vector<bool> direct_values;
   std::vector<bool> enable_beta_precision_sampling_values;
   std::vector<bool> throw_on_cholesky_error_values;

   //FIXME: these must be removed

   //old values

   std::shared_ptr<ISideInfo> Features;  // side information
   double beta_precision;
   double tol = 1e-6;
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

   const Eigen::VectorXd getMu(int n) const override;
   Eigen::MatrixXd &beta() const { return *m_beta; }

   int num_feat() const { return Features->cols(); }

private:
   void compute_Ft_y_omp(Eigen::MatrixXd& Ft_y);

public:

   void addSideInfo(const std::shared_ptr<ISideInfo>& side_info_a, double beta_precision_a, double tolerance_a, bool direct_a, bool enable_beta_precision_sampling_a, bool throw_on_cholesky_error_a);

public:
   bool save(std::shared_ptr<const StepFile> sf) const override;
   void restore(std::shared_ptr<const StepFile> sf) override;

public:
   std::ostream& info(std::ostream &os, std::string indent) override;
   std::ostream& status(std::ostream &os, std::string indent) const override;

public:
   static std::pair<double, double> posterior_beta_precision(Eigen::MatrixXd & beta, Eigen::MatrixXd & Lambda_u, double nu, double mu);
   static double sample_beta_precision(Eigen::MatrixXd & beta, Eigen::MatrixXd & Lambda_u, double nu, double mu);
};

}
