#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <memory>

#include "mvnormal.h"
#include "model.h"

#include "ILatentPrior.h"

namespace smurff {

//definition is identical except for:
//thread_vector<VectorNd> Ucol;
//thread_vector<MatrixNNd> UUcol;

//why remove update_prior method ?   

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
}