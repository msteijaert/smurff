#ifndef LATENTPRIOR_H
#define LATENTPRIOR_H

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <memory>

#include "mvnormal.h"
#include "linop.h"
#include "model.h"
#include "macau.h"

class INoiseModel;
class MacauBase;
class Macau;

 // forward declarationsc
typedef Eigen::SparseMatrix<double> SparseMatrixD;

/** interface */
class ILatentPrior {
  public:
      // c-tor
      ILatentPrior(Factors &m, int p, INoiseModel &n)
          : model(m), pos(p), U(m.U(pos)), V(m.V(pos)),
            noise(n) {} 
      virtual ~ILatentPrior() {}

      // utility
      int num_latent() const { return model.num_latent; }
      //Eigen::MatrixXd::ConstColXpr col(int i) const { return U.col(i); }
      virtual void savePriorInfo(std::string prefix) = 0;

      // work
      virtual double getLinkNorm() { return NAN; };
      virtual double getLinkLambda() { return NAN; };
      virtual bool run_slave() { return false; } // returns true if some work happened...

      virtual void sample_latents() {
          model.update_pnm(pos);
#pragma omp parallel for schedule(dynamic, 2)
          for(int n = 0; n < U.cols(); n++) sample_latent(n); 
      }

      virtual void sample_latent(int n) = 0;
      virtual Factors::PnM pnm(int n) { return model.get_pnm(pos, n); }

  private:

  protected:
      Factors &model;
      int pos;
      Eigen::MatrixXd &U, &V;
      INoiseModel &noise;
      std::vector<ILatentPrior *> siblings;
};

/** Prior without side information (pure BPMF) */

class NormalPrior : public ILatentPrior {
  public:
    NormalPrior(Factors &m, int p, INoiseModel &n);
    virtual ~NormalPrior() {}

    Eigen::VectorXd mu; 
    Eigen::MatrixXd Lambda;
    Eigen::MatrixXd WI;
    Eigen::VectorXd mu0;

    int b0;
    int df;

    virtual const Eigen::VectorXd getMu(int) const { return mu; }
    virtual const Eigen::MatrixXd getLambda(int) const { return Lambda; }

    void sample_latents() override;
    void sample_latent(int n) override;
    void savePriorInfo(std::string prefix) override;
};

class ProbitNormalPrior : public NormalPrior {
  public:
    ProbitNormalPrior(Factors &m, int p, INoiseModel &n)
        : NormalPrior(m, p, n) {}
    virtual ~ProbitNormalPrior() {}
    virtual Factors::PnM pnm(int n) { return model.get_probit_pnm(pos, n); }
};

template<class Prior>
class MasterPrior : public Prior {
  public:
    MasterPrior(Factors &m, int p, INoiseModel &n);
    virtual ~MasterPrior() {}

    virtual void sample_latents() override;
    Factors::PnM pnm(int) override;
    void addSideInfo(MacauBase &, Factors &);

  private:
    std::vector<MacauBase *> slaves;
    bool is_init = false;
};

class SlavePrior : public ILatentPrior {
  public:
    SlavePrior(Factors &m, int p, INoiseModel &n)
        : ILatentPrior(m, p, n) {}
    virtual ~SlavePrior() {}
    void sample_latents() override {};
    void sample_latent(int) override {};
    void savePriorInfo(std::string prefix) override {}

  private:
    std::vector<ILatentPrior *> siblings;
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
    MacauPrior(Factors &, int, INoiseModel &);
    virtual ~MacauPrior() {}
    void sample_latents() override;
            
    void addSideInfo(std::unique_ptr<FType> &Fmat, bool comp_FtF = false);

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
class MPIMacauPrior : public MacauPrior<FType> {
  public:
    MPIMacauPrior(SparseMF &, int, INoiseModel &);
    virtual ~MPIMacauPrior() {}

    void addSideInfo(std::unique_ptr<FType> &Fmat, bool comp_FtF = false);

    virtual void sample_beta();
    virtual bool run_slave() { sample_beta(); return true; }

    int world_rank;
    int world_size;

    int rhs() const { return rhs_for_rank[world_rank]; }

  private:
    int* rhs_for_rank = NULL;
    double* rec     = NULL;
    int* sendcounts = NULL;
    int* displs     = NULL;
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
    SpikeAndSlabPrior(Factors &, int, INoiseModel &);
    virtual ~SpikeAndSlabPrior() {}

    void savePriorInfo(std::string prefix) override;
    void sample_latents() override;
    void sample_latent(int n) override;
   
  protected:
    bool is_init = false;

};

#endif /* LATENTPRIOR_H */
