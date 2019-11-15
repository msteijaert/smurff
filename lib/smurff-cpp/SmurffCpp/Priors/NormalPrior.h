#pragma once

#include <memory>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <SmurffCpp/Utils/Distribution.h>
#include <SmurffCpp/Utils/ThreadVector.hpp>

#include <SmurffCpp/Priors/ILatentPrior.h>

namespace smurff {

// Prior without side information (pure BPMF) 
class NormalPrior : public ILatentPrior 
{
public:
  // hyperparams
  std::shared_ptr<Eigen::VectorXd> m_mu; 
  Eigen::VectorXd &hyperMu() const { return *m_mu; }

  Eigen::MatrixXd Lambda;

  // PP hyperparams
  std::shared_ptr<Eigen::MatrixXd> mu_pp; // array of size N to vector of size K 
  std::shared_ptr<Eigen::MatrixXd> Lambda_pp; // array of size N  to matrix of size K x K

  Eigen::MatrixXd WI;
  Eigen::VectorXd mu0;

  // constants
  int b0;
  int df;

protected:
   NormalPrior()
      : ILatentPrior(){}

public:
  NormalPrior(std::shared_ptr<Session> session, uint32_t mode, std::string name = "NormalPrior");
  virtual ~NormalPrior() {}
  void init() override;

  //mu in NormalPrior does not depend on column index
  //however successors of this class can override this method
  //for example in MacauPrior mu depends on Uhat.col(n)
  virtual const Eigen::VectorXd fullMu(int n) const;
  const Eigen::MatrixXd getLambda(int n) const;
  
  void sample_latent(int n) override;

  void update_prior() override;
  std::ostream &status(std::ostream &os, std::string indent) const override;
};
}
