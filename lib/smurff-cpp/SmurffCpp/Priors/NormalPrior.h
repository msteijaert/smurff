#pragma once

#include <memory>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <SmurffCpp/Utils/Distribution.h>

#include <SmurffCpp/Priors/ILatentPrior.h>

#include <SmurffCpp/model.h>

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

protected:
   NormalPrior()
      : ILatentPrior(){}

public:
  NormalPrior(std::shared_ptr<BaseSession> session, int mode, std::string name = "NormalPrior");
  virtual ~NormalPrior() {}
  void init() override;

  virtual const Eigen::VectorXd getMu(int) const;
  
  void sample_latent(int n) override;

  void update_prior() override;

  void save(std::string prefix, std::string suffix) override;
  void restore(std::string prefix, std::string suffix) override;
  virtual std::ostream &status(std::ostream &os, std::string indent) const override;

private:
   // for effiency, we keep + update Ucol and UUcol by every thread
   thread_vector<Eigen::VectorXd> Ucol;
   thread_vector<Eigen::MatrixXd> UUcol;

private:
  void initUU();
};
}