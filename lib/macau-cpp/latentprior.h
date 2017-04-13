#ifndef LATENTPRIOR_H
#define LATENTPRIOR_H

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <memory>

#include "mvnormal.h"
#include "linop.h"
#include "model.h"
#include "iface.h"

namespace Macau {

class INoiseModel;
class BaseSession;

/** interface */
class ILatentPrior {
  public:
      // c-tor
      ILatentPrior(BaseSession &m, int p, std::string name = "xxxx");
      virtual ~ILatentPrior() {}
      virtual void init() {}

      // utility
      BaseSession &sys(int s = 0);
      Factors &model(int s);
      Eigen::MatrixXd &U(int s = 0);
      Eigen::MatrixXd &V(int s = 0);
      INoiseModel &noise(int s = 0);

      int num_latent() { return Factors::num_latent; }
      int num_cols();
      int num_sys() { return sessions.size(); }

      virtual void savePriorInfo(std::string prefix) = 0;
      virtual std::ostream &printInitStatus(std::ostream &os, std::string indent);

      // work
      virtual double getLinkNorm() { return NAN; };
      virtual double getLinkLambda() { return NAN; };
      virtual bool run_slave() { return false; } // returns true if some work happened...

      virtual void sample_latents();
      virtual void sample_latent(int s, int n) = 0;
      virtual void pnm(int s, int n, VectorNd &rr, MatrixNNd &MM);

      void add(BaseSession &b);

      std::vector<BaseSession *> sessions;
      int pos;
      std::string name = "xxxx";

      thread_vector<VectorNd> rrs;
      thread_vector<MatrixNNd> MMs;

};

/** Prior without side information (pure BPMF) */

class NormalPrior : public ILatentPrior {
  public:
    NormalPrior(BaseSession &m, int p, std::string name = "NormalPrior");
    virtual ~NormalPrior() {}
    
    // updated by every thread
    thread_vector<VectorNd> Ucol;
    thread_vector<MatrixNNd> UUcol;


    // hyperparams
    Eigen::VectorXd mu; 
    Eigen::MatrixXd Lambda;
    Eigen::MatrixXd WI;
    Eigen::VectorXd mu0;

    // constants
    int b0;
    int df;

    virtual const Eigen::VectorXd getMu(int) const { return mu; }
    void sample_latents() override;
    void sample_latent(int s, int n) override;
    void savePriorInfo(std::string prefix) override;
};

template<class Prior>
class MasterPrior : public Prior {
  public:
    MasterPrior(BaseSession &m, int p);
    virtual ~MasterPrior() {}
    void init() override;

    void sample_latents() override;
    void sample_latent(int s, int n) override;
    void pnm(int, int, VectorNd&, MatrixNNd&) override;

    template<class Model>
    Model& addSlave();

    std::ostream &printInitStatus(std::ostream &os, std::string indent) override;

    double getLinkNorm() override;

  private:
    std::vector<BaseSession> slaves;
};

class SlavePrior : public ILatentPrior {
  public:
    SlavePrior(BaseSession &m, int p) : ILatentPrior(m, p, "SlavePrior") {}
    virtual ~SlavePrior() {}

    void sample_latent(int,int) override {};
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
    MacauPrior(BaseSession &m, int p);
    virtual ~MacauPrior() {}
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
    MPIMacauPrior(BaseSession &m, int p);
    virtual ~MPIMacauPrior() {}

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
    // updated by every thread
    thread_vector<VectorNd> Zcol, W2col;

    // read-only during sampling
    VectorNd Zkeep;
    ArrayNd alpha;
    VectorNd r;

    //-- hyper params
    const double prior_beta = 1; //for r
    const double prior_alpha_0 = 1.; //for alpha
    const double prior_beta_0 = 1.; //for alpha

  public:
    SpikeAndSlabPrior(BaseSession &m, int p);
    virtual ~SpikeAndSlabPrior() {}
    void init() override;

    void savePriorInfo(std::string prefix) override {}
    void sample_latents() override;
    void sample_latent(int s, int n) override;

    // mean value of Z
    double getLinkNorm() override { return Zkeep.sum(); }
};

}

#endif /* LATENTPRIOR_H */
