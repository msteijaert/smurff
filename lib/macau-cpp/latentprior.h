#ifndef LATENTPRIOR_H
#define LATENTPRIOR_H

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <memory>

#include "mvnormal.h"
#include "linop.h"
#include "model.h"

 // forward declarationsc
class FixedGaussianNoise;
class AdaptiveGaussianNoise;
class ProbitNoise;

typedef Eigen::SparseMatrix<double> SparseMatrixD;

/** interface */
class ILatentPrior {
  public:
      // c-tor
      ILatentPrior(MFactor &);

      // utility
      int num_latent() const { return fac.num_latent; }
      Eigen::MatrixXd::ConstColXpr col(int i) const { return fac.col(i); }
      virtual void saveModel(std::string prefix) = 0;

      // work
      virtual void update_prior() = 0;
      virtual double getLinkNorm() { return NAN; };
      virtual double getLinkLambda() { return NAN; };

      void sample_latents(const Eigen::MatrixXd &V);
      virtual void sample_latent(int n, const Eigen::MatrixXd &V) = 0;

  protected:
      MFactor &fac;
};

/** Prior without side information (pure BPMF) */
template<class NoiseModel>
class NormalPrior : public ILatentPrior {
  public:
    Eigen::VectorXd mu; 
    Eigen::MatrixXd Lambda;
    Eigen::MatrixXd WI;
    Eigen::VectorXd mu0;

    int b0;
    int df;

    NoiseModel &noise;

  public:
    NormalPrior(MFactor &d, NoiseModel &noise);

    void init();
    void update_prior() override;
    void saveModel(std::string prefix) override;

    virtual const Eigen::VectorXd getMu(int) const { return mu; }
    virtual const Eigen::VectorXd getLambda(int) const { return Lambda; }

    virtual void sample_latent(int n, const Eigen::MatrixXd &V) override;
};

/** Prior with side information */
template<class FType, class NoiseModel>
class MacauPrior : public NormalPrior<NoiseModel> {
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
    MacauPrior(MFactor &d, NoiseModel &noise, std::unique_ptr<FType> &Fmat, bool comp_FtF);

    void init(const int num_latent, std::unique_ptr<FType> &Fmat, bool comp_FtF);

    void update_prior() override;
    double getLinkNorm() override;
    double getLinkLambda() override { return lambda_beta; };
    const Eigen::VectorXd getMu(int n) const override { return this->mu + Uhat.col(n); }
    const Eigen::VectorXd getLambda(int) const override { return this->Lambda; }

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
