#pragma once

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

/* macau
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <memory>
#include "mvnormal.h"
#include "linop.h"
#include "sparsetensor.h"

 // forward declarations
class FixedGaussianNoise;
class AdaptiveGaussianNoise;
class ProbitNoise;
class MatrixData;

// interface
class ILatentPrior {
  public:
    virtual void sample_latents(Eigen::MatrixXd &U, const Eigen::SparseMatrix<double> &mat, double mean_value,
                        const Eigen::MatrixXd &samples, double alpha, const int num_latent) = 0;
    virtual void sample_latents(FixedGaussianNoise& noise, Eigen::MatrixXd &U, const Eigen::SparseMatrix<double> &mat,
                        double mean_value, const Eigen::MatrixXd &samples, const int num_latent);
    virtual void sample_latents(AdaptiveGaussianNoise& noise, Eigen::MatrixXd &U, const Eigen::SparseMatrix<double> &mat,
                        double mean_value, const Eigen::MatrixXd &samples, const int num_latent);
    virtual void sample_latents(ProbitNoise& noise, Eigen::MatrixXd &U, const Eigen::SparseMatrix<double> &mat,
                        double mean_value, const Eigen::MatrixXd &samples, const int num_latent) = 0;
    // general functions (called from outside)
    void sample_latents(FixedGaussianNoise& noiseModel, MatrixData & matrixData,
                                std::vector< std::unique_ptr<Eigen::MatrixXd> > & samples, int mode, const int num_latent);
    void sample_latents(AdaptiveGaussianNoise& noiseModel, MatrixData & matrixData,
                                std::vector< std::unique_ptr<Eigen::MatrixXd> > & samples, int mode, const int num_latent);
    void sample_latents(ProbitNoise& noiseModel, MatrixData & matrixData,
                                std::vector< std::unique_ptr<Eigen::MatrixXd> > & samples, int mode, const int num_latent);
    // for tensor
    void sample_latents(FixedGaussianNoise& noiseModel, TensorData & data,
                                std::vector< std::unique_ptr<Eigen::MatrixXd> > & samples, int mode, const int num_latent);
    void sample_latents(AdaptiveGaussianNoise& noiseModel, TensorData & data,
                                std::vector< std::unique_ptr<Eigen::MatrixXd> > & samples, int mode, const int num_latent);
    virtual void sample_latents(ProbitNoise& noiseModel, TensorData & data,
                                std::vector< std::unique_ptr<Eigen::MatrixXd> > & samples, int mode, const int num_latent) = 0;
    virtual void sample_latents(double noisePrecision, TensorData & data,
                                std::vector< std::unique_ptr<Eigen::MatrixXd> > & samples, int mode, const int num_latent) = 0;

    void virtual update_prior(const Eigen::MatrixXd &U) {};
    virtual double getLinkNorm() { return NAN; };
    virtual double getLinkLambda() { return NAN; };
    virtual void saveModel(std::string prefix) {};
    virtual ~ILatentPrior() {};
};


// Prior without side information (pure BPMF)
class BPMFPrior : public ILatentPrior {
  public:
    Eigen::VectorXd mu; 
    Eigen::MatrixXd Lambda;
    Eigen::MatrixXd WI;
    Eigen::VectorXd mu0;

    int b0;
    int df;

  public:
    BPMFPrior(const int nlatent) { init(nlatent); }
    BPMFPrior() : BPMFPrior(10) {}
    void init(const int num_latent);

    void sample_latents(Eigen::MatrixXd &U, const Eigen::SparseMatrix<double> &mat, double mean_value,
                                   const Eigen::MatrixXd &samples, double alpha, const int num_latent) override;
    void sample_latents(ProbitNoise & noise, Eigen::MatrixXd &U, const Eigen::SparseMatrix<double> &mat,
                                   double mean_value, const Eigen::MatrixXd &samples, const int num_latent) override;

    void sample_latents(ProbitNoise& noiseModel, TensorData & data,
                                std::vector< std::unique_ptr<Eigen::MatrixXd> > & samples, int mode, const int num_latent) override;
    void sample_latents(double noisePrecision, TensorData & data,
                                std::vector< std::unique_ptr<Eigen::MatrixXd> > & samples, int mode, const int num_latent) override;
    void update_prior(const Eigen::MatrixXd &U) override;
    void saveModel(std::string prefix) override;
};

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

std::pair<double,double> posterior_lambda_beta(Eigen::MatrixXd & beta, Eigen::MatrixXd & Lambda_u, double nu, double mu);
double sample_lambda_beta(Eigen::MatrixXd & beta, Eigen::MatrixXd & Lambda_u, double nu, double mu);

void sample_latent(Eigen::MatrixXd &s,
                   int mm,
                   const Eigen::SparseMatrix<double> &mat,
                   double mean_rating,
                   const Eigen::MatrixXd &samples,
                   double alpha,
                   const Eigen::VectorXd &mu_u,
                   const Eigen::MatrixXd &Lambda_u,
                   const int num_latent);

void sample_latent_blas(Eigen::MatrixXd &s,
                        int mm,
                        const Eigen::SparseMatrix<double> &mat,
                        double mean_rating,
                        const Eigen::MatrixXd &samples,
                        double alpha,
                        const Eigen::VectorXd &mu_u,
                        const Eigen::MatrixXd &Lambda_u,
                        const int num_latent);

void sample_latent_blas_probit(Eigen::MatrixXd &s,
                        int mm,
                        const Eigen::SparseMatrix<double> &mat,
                        double mean_rating,
                        const Eigen::MatrixXd &samples,
                        const Eigen::VectorXd &mu_u,
                        const Eigen::MatrixXd &Lambda_u,
                        const int num_latent);
void sample_latent_tensor(std::unique_ptr<Eigen::MatrixXd> &U,
                          int n,
                          std::unique_ptr<SparseMode> & sparseMode,
                          VectorView<Eigen::MatrixXd> & view,
                          double mean_value,
                          double alpha,
                          Eigen::VectorXd & mu,
                          Eigen::MatrixXd & Lambda);
Eigen::MatrixXd A_mul_B(Eigen::MatrixXd & A, Eigen::MatrixXd & B);
Eigen::MatrixXd A_mul_B(Eigen::MatrixXd & A, SparseFeat & B);
Eigen::MatrixXd A_mul_B(Eigen::MatrixXd & A, SparseDoubleFeat & B);

MacauPrior<Eigen::MatrixXd>* make_dense_prior(int nlatent, double* ptr, int nrows, int ncols, bool colMajor, bool comp_FtF);
*/