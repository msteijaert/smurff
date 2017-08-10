#ifndef LATENTPRIOR_H
#define LATENTPRIOR_H

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <memory>

#include "mvnormal.h"
#include "linop.h"
#include "model.h"
#include "session.h"

namespace smurff {

class INoiseModel;
class BaseSession;

/** interface */
class ILatentPrior {
  public:
      // c-tor
      ILatentPrior(BaseSession &s, int m, std::string name = "xxxx");
      virtual ~ILatentPrior() {}
      virtual void init();

      // utility
      Model &model() const;
      Data  &data() const;
      INoiseModel &noise();
      Eigen::MatrixXd &U();
      Eigen::MatrixXd &V();
      int num_latent() const { return model().nlatent(); }
      int num_cols()   const { return model().U(mode).cols(); }

      virtual void save(std::string prefix, std::string suffix) = 0;
      virtual void restore(std::string prefix, std::string suffix) = 0;
      virtual std::ostream &info(std::ostream &os, std::string indent);
      virtual std::ostream &status(std::ostream &os, std::string indent) const = 0;

      // work
      virtual bool run_slave() { return false; } // returns true if some work happened...

      virtual void sample_latents();
      virtual void sample_latent(int n) = 0;

      void add(BaseSession &b);

      BaseSession &session;
      int mode;
      std::string name = "xxxx";

      thread_vector<VectorNd> rrs;
      thread_vector<MatrixNNd> MMs;

};

/** Prior without side information (pure BPMF) */

class NormalPrior : public ILatentPrior {
  public:
    NormalPrior(BaseSession &m, int p, std::string name = "NormalPrior");
    virtual ~NormalPrior() {}
    void init() override;

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
    void sample_latent(int n) override;
    void save(std::string prefix, std::string suffix) override;
    void restore(std::string prefix, std::string suffix) override;
    virtual std::ostream &status(std::ostream &os, std::string indent) const override;

  private:
    // for effiency, we keep + update Ucol and UUcol by every thread
    thread_vector<VectorNd> Ucol;
    thread_vector<MatrixNNd> UUcol;
    void initUU();
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

    double getLinkLambda() { return lambda_beta; };
    const Eigen::VectorXd getMu(int n) const override { return this->mu + Uhat.col(n); }

    void compute_Ft_y_omp(Eigen::MatrixXd &);
    virtual void sample_beta();
    void setLambdaBeta(double lb) { lambda_beta = lb; };
    void setTol(double t) { tol = t; };
    void save(std::string prefix, std::string suffix) override;
    void restore(std::string prefix, std::string suffix) override;
    std::ostream &info(std::ostream &os, std::string indent) override;
    std::ostream &status(std::ostream &os, std::string indent) const override;

  private:
    void sample_beta_direct();
    void sample_beta_cg();
};


/** Prior with side information */
template<class FType>
class MPIMacauPrior : public MacauPrior<FType> {
  public:
    MPIMacauPrior(BaseSession &m, int p);
    virtual ~MPIMacauPrior() {}

    void addSideInfo(std::unique_ptr<FType> &Fmat, bool comp_FtF = false);
    std::ostream &info(std::ostream &os, std::string indent) override;

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
    thread_vector<MatrixNNd> Zcol, W2col;

    // read-only during sampling
    MatrixNNd Zkeep;
    ArrayNNd alpha;
    MatrixNNd r;

    //-- hyper params
    const double prior_beta = 1; //for r
    const double prior_alpha_0 = 1.; //for alpha
    const double prior_beta_0 = 1.; //for alpha

  public:
    SpikeAndSlabPrior(BaseSession &m, int p);
    virtual ~SpikeAndSlabPrior() {}
    void init() override;

    void save(std::string prefix, std::string suffix) override {}
    void restore(std::string prefix, std::string suffix) override {}
    void sample_latents() override;
    void sample_latent(int n) override;

    // mean value of Z
    std::ostream &status(std::ostream &os, std::string indent) const override;
};

}

#endif /* LATENTPRIOR_H */
