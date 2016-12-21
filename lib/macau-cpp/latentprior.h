#ifndef LATENTPRIOR_H
#define LATENTPRIOR_H

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <memory>
#include "mvnormal.h"
#include "linop.h"
#include "noisemodels.h"

 // forward declarations
class FixedGaussianNoise;
class AdaptiveGaussianNoise;
class ProbitNoise;

typedef Eigen::SparseMatrix<double> SparseMatrixD;

/** interface */
class ILatentPrior {
  public:
      // c-tor
      ILatentPrior(SparseMatrixD &Y, int nlatent);

      // data
      Eigen::MatrixXd U;
      SparseMatrixD &Y;
      double mean_value;

      // utility
      int num_latent() const { return U.rows(); }
      Eigen::MatrixXd::ColXpr col(int i) { return U.col(i); }
      virtual const Eigen::VectorXd getMu(int n) const = 0;
      virtual const Eigen::VectorXd getLambda(int) const = 0;
      virtual void saveModel(std::string prefix) = 0;

      // work
      virtual void update_prior() = 0;
      virtual double getLinkNorm() { return NAN; };
      virtual double getLinkLambda() { return NAN; };

      template<class NoiseModel>
      void sample_latents(const Eigen::MatrixXd &V, NoiseModel &noise);

      template<class NoiseModel>
      void sample_latent(int n, const Eigen::MatrixXd &V, NoiseModel &noise);
};

/** Prior without side information (pure BPMF) */
class BPMFPrior : public ILatentPrior {
  public:
    Eigen::VectorXd mu; 
    Eigen::MatrixXd Lambda;
    Eigen::MatrixXd WI;
    Eigen::VectorXd mu0;

    int b0;
    int df;

  public:
    BPMFPrior(SparseMatrixD &Y, const int nlatent = 10) 
       : ILatentPrior(Y,nlatent) { init(nlatent); }

    void init(int);
    void update_prior() override;
    void saveModel(std::string prefix) override;

    const Eigen::VectorXd getMu(int) const override { return mu; }
    const Eigen::VectorXd getLambda(int) const override { return Lambda; }
};

/** Prior without side information (pure BPMF) */
template<class FType>
class MacauPrior : public ILatentPrior {
  public:
    Eigen::MatrixXd Uhat;
    std::unique_ptr<FType> F;  /* side information */
    Eigen::MatrixXd FtF;       /* F'F */
    Eigen::MatrixXd beta;      /* link matrix */
    bool use_FtF;
    double lambda_beta;
    double lambda_beta_mu0; /* Hyper-prior for lambda_beta */
    double lambda_beta_nu0; /* Hyper-prior for lambda_beta */

    Eigen::VectorXd mu; 
    Eigen::MatrixXd Lambda;
    Eigen::MatrixXd WI;
    Eigen::VectorXd mu0;

    int b0;
    int df;

    double tol = 1e-6;

  public:
    MacauPrior(SparseMatrixD &Y, const int nlatent, std::unique_ptr<FType> &Fmat, bool comp_FtF) 
       : ILatentPrior(Y,nlatent) 
    {
        init(nlatent, Fmat, comp_FtF);
    }

    void init(const int num_latent, std::unique_ptr<FType> &Fmat, bool comp_FtF);

    void update_prior() override;
    double getLinkNorm() override;
    double getLinkLambda() override { return lambda_beta; };
    const Eigen::VectorXd getMu(int n) const override { return mu + Uhat.col(n); }
    const Eigen::VectorXd getLambda(int) const override { return Lambda; }

    void sample_beta();
    void setLambdaBeta(double lb) { lambda_beta = lb; };
    void setTol(double t) { tol = t; };
    void saveModel(std::string prefix) override;
};

std::pair<double,double> posterior_lambda_beta(Eigen::MatrixXd & beta, Eigen::MatrixXd & Lambda_u, double nu, double mu);
double sample_lambda_beta(Eigen::MatrixXd & beta, Eigen::MatrixXd & Lambda_u, double nu, double mu);

Eigen::MatrixXd A_mul_B(Eigen::MatrixXd & A, Eigen::MatrixXd & B);
Eigen::MatrixXd A_mul_B(Eigen::MatrixXd & A, SparseFeat & B);
Eigen::MatrixXd A_mul_B(Eigen::MatrixXd & A, SparseDoubleFeat & B);

#endif /* LATENTPRIOR_H */
