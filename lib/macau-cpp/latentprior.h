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

/** interface */
class ILatentPrior {
  public:
      // c-tor
      ILatentPrior(MacauBase &m, int p, std::string name = "xxxx");
      virtual ~ILatentPrior() {}
      virtual void init() {}

      // utility
      Factors &model() const;
      int num_latent() const;
      INoiseModel &noise() const;
      virtual void savePriorInfo(std::string prefix) = 0;
      virtual std::ostream &printInitStatus(std::ostream &os, std::string indent);

      // work
      virtual double getLinkNorm() { return NAN; };
      virtual double getLinkLambda() { return NAN; };
      virtual bool run_slave() { return false; } // returns true if some work happened...

      virtual void sample_latents() {
          model().update_pnm(pos);
#pragma omp parallel for
          for(int n = 0; n < U.cols(); n++) {
#pragma omp task shared(n)
              sample_latent(n); 
          }
      }

      virtual void sample_latent(int n) = 0;
      virtual void pnm(int n, VectorNd &rr, MatrixNNd &MM) { model().get_pnm(pos, n, rr, MM); }

      virtual void addSibling(MacauBase &b) = 0;
      template<class Prior>
      void addSiblingTempl(MacauBase &b);

      MacauBase &macau;
      int pos;
      Eigen::MatrixXd &U, &V;
      std::vector<ILatentPrior *> siblings;
      std::string name = "xxxx";

      thread_vector<VectorNd> rrs;
      thread_vector<MatrixNNd> MMs;

};

/** Prior without side information (pure BPMF) */

class NormalPrior : public ILatentPrior {
  public:
    NormalPrior(MacauBase &m, int p, std::string name = "NormalPrior");
    virtual ~NormalPrior() {}
    void addSibling(MacauBase &b) override;

    Eigen::VectorXd mu; 
    Eigen::MatrixXd Lambda;
    Eigen::MatrixXd WI;
    Eigen::VectorXd mu0;

    int b0;
    int df;

    virtual const Eigen::VectorXd getMu(int) const { return mu; }
    void sample_latents() override;
    void sample_latent(int n) override;
    void savePriorInfo(std::string prefix) override;
};

class ProbitNormalPrior : public NormalPrior {
  public:
    ProbitNormalPrior(MacauBase &m, int p)
        : NormalPrior(m, p) {}
    virtual ~ProbitNormalPrior() {}
    virtual void pnm(int n, VectorNd &rr, MatrixNNd &MM) { model().get_probit_pnm(pos, n, rr, MM); }
};

template<class Prior>
class MasterPrior : public Prior {
  public:
    MasterPrior(MacauBase &m, int p);
    virtual ~MasterPrior() {}
    void addSibling(MacauBase &) override { assert(false); }
    void init() override;

    virtual void sample_latents() override;
    void pnm(int,VectorNd&,MatrixNNd&) override;

    template<class Model>
    Model& addSlave();

    std::ostream &printInitStatus(std::ostream &os, std::string indent) override;

  private:
    std::vector<MacauBase> slaves;
};

class SlavePrior : public ILatentPrior {
  public:
    SlavePrior(MacauBase &m, int p) : ILatentPrior(m, p, "SlavePrior") {}
    virtual ~SlavePrior() {}
    void addSibling(MacauBase &) override { assert(false); }

    void sample_latent(int) override {};
    void savePriorInfo(std::string prefix) override {}
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
    MacauPrior(MacauBase &m, int p);
    virtual ~MacauPrior() {}
    void addSibling(MacauBase &b) override;
    void init() override;

    void sample_latents() override;
            
    void addSideInfo(std::unique_ptr<FType> &Fmat, bool comp_FtF = false);

    double getLinkNorm() override;
    double getLinkLambda() override { return lambda_beta; };
    const Eigen::VectorXd getMu(int n) const override { return this->mu + Uhat.col(n); }

    void compute_Ft_y_omp(Eigen::MatrixXd &);
    virtual void sample_beta();
    void setLambdaBeta(double lb) { lambda_beta = lb; };
    void setTol(double t) { tol = t; };
    void savePriorInfo(std::string prefix) override;
    std::ostream &printInitStatus(std::ostream &os, std::string indent) override;
};


/** Prior with side information */
template<class FType>
class MPIMacauPrior : public MacauPrior<FType> {
  public:
    MPIMacauPrior(MacauBase &m, int p);
    virtual ~MPIMacauPrior() {}
    void addSibling(MacauBase &b) override;

    void addSideInfo(std::unique_ptr<FType> &Fmat, bool comp_FtF = false);
    std::ostream &printInitStatus(std::ostream &os, std::string indent) override;

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
    SpikeAndSlabPrior(MacauBase &m, int p);
    virtual ~SpikeAndSlabPrior() {}
    void init() override;
    void addSibling(MacauBase &b) override;

    void savePriorInfo(std::string prefix) override {}
    void sample_latents() override;
    void sample_latent(int n) override;
};

#endif /* LATENTPRIOR_H */
