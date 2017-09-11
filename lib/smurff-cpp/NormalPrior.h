#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <memory>

#include "mvnormal.h"
#include "linop.h"
#include "model.h"

#include "ILatentPrior.h"

namespace smurff {

// Prior without side information (pure BPMF) 
class NormalPrior : public ILatentPrior 
{
public:
  // hyperparams
  Eigen::VectorXd mu; 
  Eigen::MatrixXd Lambda;
  Eigen::MatrixXd WI;
  Eigen::VectorXd mu0;

  // constants
  int b0;
  int df;

public:
  NormalPrior(BaseSession &m, int p, std::string name = "NormalPrior");
  virtual ~NormalPrior() {}
  void init() override;

  virtual const Eigen::VectorXd getMu(int) const;
  void sample_latents() override;
  void sample_latent(int n) override;
  void save(std::string prefix, std::string suffix) override;
  void restore(std::string prefix, std::string suffix) override;
  virtual std::ostream &status(std::ostream &os, std::string indent) const override;

private:
   // for effiency, we keep + update Ucol and UUcol by every thread
   thread_vector<VectorNd> Ucol;
   thread_vector<MatrixNNd> UUcol;

private:
  void initUU();
};

//macau
/*
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
*/
}