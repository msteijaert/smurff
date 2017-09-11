#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <memory>

#include "mvnormal.h"
#include "linop.h"
#include "model.h"

#include "NormalPrior.h"

namespace smurff {

std::pair<double,double> posterior_lambda_beta(Eigen::MatrixXd & beta, Eigen::MatrixXd & Lambda_u, double nu, double mu);
double sample_lambda_beta(Eigen::MatrixXd & beta, Eigen::MatrixXd & Lambda_u, double nu, double mu);

/// Prior with side information
template<class FType>
class MacauPrior : public NormalPrior 
{
  public:
    Eigen::MatrixXd Uhat;
    std::unique_ptr<FType> F;  /* side information */
    Eigen::MatrixXd FtF;       /* F'F */
    Eigen::MatrixXd beta;      /* link matrix */
    bool use_FtF;
    double lambda_beta;
    double lambda_beta_mu0; /* Hyper-prior for lambda_beta */
    double lambda_beta_nu0; /* Hyper-prior for lambda_beta */

    double tol = 1e-6;

  public:
    MacauPrior(BaseSession &m, int p);
    virtual ~MacauPrior() {}
    void init() override;

    void sample_latents() override;
            
    void addSideInfo(std::unique_ptr<FType> &Fmat, bool comp_FtF = false);

    double getLinkLambda();
    const Eigen::VectorXd getMu(int n) const override;

    void compute_Ft_y_omp(Eigen::MatrixXd &);
    virtual void sample_beta();
    void setLambdaBeta(double lb);
    void setTol(double t);
    void save(std::string prefix, std::string suffix) override;
    void restore(std::string prefix, std::string suffix) override;
    std::ostream &info(std::ostream &os, std::string indent) override;
    std::ostream &status(std::ostream &os, std::string indent) const override;

  private:
    void sample_beta_direct();
    void sample_beta_cg();
};

//macau
/*
std::pair<double,double> posterior_lambda_beta(Eigen::MatrixXd & beta, Eigen::MatrixXd & Lambda_u, double nu, double mu);
double sample_lambda_beta(Eigen::MatrixXd & beta, Eigen::MatrixXd & Lambda_u, double nu, double mu);

// Prior without side information (pure BPMF)
template<class FType>
class MacauPrior : public ILatentPrior {
  public:
    Eigen::MatrixXd Uhat;
    std::unique_ptr<FType> F;  // side information
    Eigen::MatrixXd FtF;       // F'F
    Eigen::MatrixXd beta;      // link matrix
    bool use_FtF;
    double lambda_beta;
    double lambda_beta_mu0; // Hyper-prior for lambda_beta
    double lambda_beta_nu0; // Hyper-prior for lambda_beta

    Eigen::VectorXd mu; 
    Eigen::MatrixXd Lambda;
    Eigen::MatrixXd WI;
    Eigen::VectorXd mu0;

    int b0;
    int df;

    double tol = 1e-6;

  public:
    MacauPrior(const int nlatent, std::unique_ptr<FType> &Fmat, bool comp_FtF) { init(nlatent, Fmat, comp_FtF); }
    MacauPrior() {}

    void init(const int num_latent, std::unique_ptr<FType> &Fmat, bool comp_FtF);

    void sample_latents(Eigen::MatrixXd &U, const Eigen::SparseMatrix<double> &mat, double mean_value,
                                   const Eigen::MatrixXd &samples, double alpha, const int num_latent) override;
    void sample_latents(ProbitNoise & noise, Eigen::MatrixXd &U, const Eigen::SparseMatrix<double> &mat,
                                   double mean_value, const Eigen::MatrixXd &samples, const int num_latent) override;
    void sample_latents(ProbitNoise& noiseModel, TensorData & data,
                                std::vector< std::unique_ptr<Eigen::MatrixXd> > & samples, int mode, const int num_latent) override;
    void sample_latents(double noisePrecision, TensorData & data,
                                std::vector< std::unique_ptr<Eigen::MatrixXd> > & samples, int mode, const int num_latent) override;
    void update_prior(const Eigen::MatrixXd &U) override;
    double getLinkNorm();
    double getLinkLambda() { return lambda_beta; };
    void sample_beta(const Eigen::MatrixXd &U);
    void setLambdaBeta(double lb) { lambda_beta = lb; };
    void setTol(double t) { tol = t; };
    void saveModel(std::string prefix) override;
};
*/

}