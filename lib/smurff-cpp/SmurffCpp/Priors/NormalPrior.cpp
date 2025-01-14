#include "NormalPrior.h"

#include <iomanip>

#include <SmurffCpp/Utils/linop.h>
#include <SmurffCpp/IO/MatrixIO.h>
#include <SmurffCpp/Utils/counters.h>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <SmurffCpp/Utils/Distribution.h>
#include <SmurffCpp/Utils/Error.h>

using namespace smurff;

//  base class NormalPrior

NormalPrior::NormalPrior(std::shared_ptr<Session> session, uint32_t mode, std::string name)
   : ILatentPrior(session, mode, name)
{

}

void NormalPrior::init()
{
   //does not look that there was such init previously
   ILatentPrior::init();

   const int K = num_latent();
   m_mu = std::make_shared<Eigen::VectorXd>(K);
   hyperMu().setZero();

   Lambda.resize(K, K);
   Lambda.setIdentity();
   Lambda *= 10;

   // parameters of Inv-Whishart distribution
   WI.resize(K, K);
   WI.setIdentity();
   mu0.resize(K);
   mu0.setZero();
   b0 = 2;
   df = K;

   const auto &config = getSession().getConfig();
   if (config.hasPropagatedPosterior(getMode()))
   {
      mu_pp = std::make_shared<Eigen::MatrixXd>(matrix_utils::dense_to_eigen(*config.getMuPropagatedPosterior(getMode())));
      Lambda_pp = std::make_shared<Eigen::MatrixXd>(matrix_utils::dense_to_eigen(*config.getLambdaPropagatedPosterior(getMode())));
      m_name += " with posterior propagation";
   }
}

const Eigen::VectorXd NormalPrior::fullMu(int n) const
{
   if (getSession().getConfig().hasPropagatedPosterior(getMode()))
   {
      return mu_pp->col(n);
   }
   //else
   return hyperMu();
}

const Eigen::MatrixXd NormalPrior::getLambda(int n) const
{
   if (getSession().getConfig().hasPropagatedPosterior(getMode()))
   {
      return Eigen::Map<Eigen::MatrixXd>(Lambda_pp->col(n).data(), num_latent(), num_latent());
   }
   //else
   return Lambda;
}
void NormalPrior::update_prior()
{
   std::tie(hyperMu(), Lambda) = CondNormalWishart(num_item(), getUUsum(), getUsum(), mu0, b0, WI, df);
}

//n is an index of column in U matrix
void  NormalPrior::sample_latent(int n)
{
   const auto &mu_u = fullMu(n);
   const auto &Lambda_u = getLambda(n);

   Eigen::VectorXd &rr = rrs.local();
   Eigen::MatrixXd &MM = MMs.local();

   rr.setZero();
   MM.setZero();

   // add pnm
   data().getMuLambda(model(), m_mode, n, rr, MM);

   // add hyperparams
   rr.noalias() += Lambda_u * mu_u;
   MM.noalias() += Lambda_u;

   //Solve system of linear equations for x: MM * x = rr - not exactly correct  because we have random part
   //Sample from multivariate normal distribution with mean rr and precision matrix MM

   Eigen::LLT<Eigen::MatrixXd> chol;
   {
      chol = MM.llt(); // compute the Cholesky decomposition X = L * U
      if(chol.info() != Eigen::Success)
      {
         THROWERROR("Cholesky Decomposition failed!");
      }
   }

   chol.matrixL().solveInPlace(rr); // solve for y: y = L^-1 * b
   rr.noalias() += nrandn(num_latent());
   chol.matrixU().solveInPlace(rr); // solve for x: x = U^-1 * y
   
   U().col(n).noalias() = rr; // rr is equal to x
}

std::ostream &NormalPrior::status(std::ostream &os, std::string indent) const
{
   os << indent << m_name << std::endl;
   os << indent << "  mu: " <<  hyperMu().transpose() << std::endl;
   return os;
}
