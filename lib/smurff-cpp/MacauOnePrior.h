#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <memory>

#include "ILatentPrior.h"

namespace smurff {

//definition is the same except that now:
// int num_latent; is not stored in this class
// Eigen::SparseMatrix<double> &Yc; is stored in this class

//also why remove init method and put everything in constructor if we have 
//init method in other priors and the other method addSideInfo which we use in pair

//why remove update_prior method ?

template<class FType>
class MacauOnePrior : public ILatentPrior 
{
// smurff variables
public:
   Eigen::MatrixXd Uhat;

   std::unique_ptr<FType> F;  /* side information */
   Eigen::VectorXd F_colsq;   // sum-of-squares for every feature (column)

   Eigen::MatrixXd beta;      /* link matrix */
   Eigen::VectorXd lambda_beta;
   double lambda_beta_a0; /* Hyper-prior for lambda_beta */
   double lambda_beta_b0; /* Hyper-prior for lambda_beta */

   Eigen::VectorXd mu;
   Eigen::VectorXd lambda;
   double lambda_a0;
   double lambda_b0;

   int l0;

   Eigen::SparseMatrix<double> &Yc;

// smurff methods
public:
   MacauOnePrior(BaseSession &, int); 
   void addSideInfo(std::unique_ptr<FType> &Fmat, bool);
   
   void sample_latent(int) override;
   void sample_latents() override;
   double getLinkLambda();
   
   void sample_beta(const Eigen::MatrixXd &U);
   void sample_mu_lambda(const Eigen::MatrixXd &U);
   void sample_lambda_beta();
   void setLambdaBeta(double lb);

   void save(std::string prefix, std::string suffix) override;
   void restore(std::string prefix, std::string suffix) override;
   std::ostream &status(std::ostream &os, std::string indent) const override;
};
}