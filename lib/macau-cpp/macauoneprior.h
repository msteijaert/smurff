#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <memory>
#include "latentprior.h"

template<class FType>
class MacauOnePrior : public ILatentPrior {
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

  public:
    MacauOnePrior(MacauBase &, int); 
    void addSibling(MacauBase &) override;
    void addSideInfo(std::unique_ptr<FType> &Fmat, bool);
    
    void sample_latent(int) override;
    void sample_latents() override;
    double getLinkNorm() override { return beta.norm(); };
    double getLinkLambda() override { return lambda_beta.mean(); };
    void sample_beta(const Eigen::MatrixXd &U);
    void sample_mu_lambda(const Eigen::MatrixXd &U);
    void sample_lambda_beta();
    void setLambdaBeta(double lb) { lambda_beta = Eigen::VectorXd::Constant(this->num_latent(), lb); };
    void savePriorInfo(std::string prefix) override;

     void pnm(int,VectorNd &, MatrixNNd &) override;
};

