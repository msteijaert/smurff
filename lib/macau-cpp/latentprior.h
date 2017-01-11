#ifndef LATENTPRIOR_H
#define LATENTPRIOR_H

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <memory>

#include "mvnormal.h"
#include "linop.h"
#include "model.h"

class INoiseModel;

 // forward declarationsc
typedef Eigen::SparseMatrix<double> SparseMatrixD;

/** interface */
class ILatentPrior {
  public:
      // c-tor
      ILatentPrior(MFactor &, INoiseModel &);

      // utility
      int num_latent() const { return fac.num_latent; }
      Eigen::MatrixXd::ConstColXpr col(int i) const { return fac.col(i); }
      virtual void savePriorInfo(std::string prefix) = 0;

      // work
      virtual void update_prior() = 0;
      virtual double getLinkNorm() { return NAN; };
      virtual double getLinkLambda() { return NAN; };
      virtual bool run_slave() { return false; } // returns true if some work happened...

      void sample_latents(const Eigen::MatrixXd &V);
      virtual void sample_latent(int n, const Eigen::MatrixXd &V) = 0;

  protected:
      MFactor &fac;
      INoiseModel &noise;
};

/** Prior without side information (pure BPMF) */
class NormalPrior : public ILatentPrior {
  public:
    Eigen::VectorXd mu; 
    Eigen::MatrixXd Lambda;
    Eigen::MatrixXd WI;
    Eigen::VectorXd mu0;

    int b0;
    int df;


  public:
    NormalPrior(MFactor &d, INoiseModel &noise);

    void update_prior() override;
    void savePriorInfo(std::string prefix) override;

    virtual const Eigen::VectorXd getMu(int) const { return mu; }
    virtual const Eigen::MatrixXd getLambda(int) const { return Lambda; }

    void sample_latent(int n, const Eigen::MatrixXd &V) override;
};

/** Prior with side information */
template<class FType>
class MacauPrior : public NormalPrior {
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
    MacauPrior(MFactor &d, INoiseModel &noise);
            
    void addSideInfo(std::unique_ptr<FType> &Fmat, bool comp_FtF = false);

    void update_prior() override;
    double getLinkNorm() override;
    double getLinkLambda() override { return lambda_beta; };
    const Eigen::VectorXd getMu(int n) const override { return this->mu + Uhat.col(n); }
    const Eigen::MatrixXd getLambda(int) const override { return this->Lambda; }

    void compute_Ft_y_omp(Eigen::MatrixXd &);
    virtual void sample_beta();
    void setLambdaBeta(double lb) { lambda_beta = lb; };
    void setTol(double t) { tol = t; };
    void savePriorInfo(std::string prefix) override;
};


/** Prior with side information */
template<class FType>
class MacauMPIPrior : public MacauPrior<FType> {
  public:
    MacauMPIPrior(MFactor &d, INoiseModel &noise, int world_rank) 
        : MacauPrior<FType>(d, noise), world_rank(world_rank) {}

    virtual void sample_beta();
    virtual bool run_slave() { sample_beta(); return true; }
  private:
    int world_rank;
};


typedef Eigen::VectorXd VectorNd;
typedef Eigen::MatrixXd MatrixNNd;
typedef Eigen::ArrayXd ArrayNd;

/** Spike and slab prior */
class SpikeAndSlabPrior : public ILatentPrior {
  public:
    VectorNd Zcol, W2col, Zkeep;
    ArrayNd alpha;
    VectorNd r;

    //-- hyper params
    const double prior_beta = 1; //for r
    const double prior_alpha_0 = 1.; //for alpha
    const double prior_beta_0 = 1.; //for alpha

    
 
  public:
    SpikeAndSlabPrior(MFactor &d, INoiseModel &noise);
    void update_prior() override;
    void savePriorInfo(std::string prefix) override;
    void sample_latent(int n, const Eigen::MatrixXd &V) override;
};

std::pair<double,double> posterior_lambda_beta(Eigen::MatrixXd & beta, Eigen::MatrixXd & Lambda_u, double nu, double mu);
double sample_lambda_beta(Eigen::MatrixXd & beta, Eigen::MatrixXd & Lambda_u, double nu, double mu);

Eigen::MatrixXd A_mul_B(Eigen::MatrixXd & A, Eigen::MatrixXd & B);
Eigen::MatrixXd A_mul_B(Eigen::MatrixXd & A, SparseFeat & B);
Eigen::MatrixXd A_mul_B(Eigen::MatrixXd & A, SparseDoubleFeat & B);

#endif /* LATENTPRIOR_H */
