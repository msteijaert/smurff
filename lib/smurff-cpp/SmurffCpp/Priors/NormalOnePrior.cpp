#include "NormalOnePrior.h"
#include <SmurffCpp/IO/MatrixIO.h>

using namespace smurff;
using namespace Eigen;

NormalOnePrior::NormalOnePrior(std::shared_ptr<BaseSession> session, uint32_t mode)
   : ILatentPrior(session, mode, "NormalOnePrior")
{

}

void NormalOnePrior::init()
{
   //does not look that there was such init previously
   ILatentPrior::init();

   const int K = num_latent();
   mu.resize(K);
   mu.setZero();

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
}

void NormalOnePrior::update_prior()
{
    const int N = num_cols();
    const auto cov = *U() * U()->transpose();
    const auto sum = U()->rowwise().sum();
    std::tie(mu, Lambda) = CondNormalWishart(N, cov, sum, mu0, b0, WI, df);
}

void NormalOnePrior::sample_latent(int d)
{
   const int K = num_latent();

   VectorXd Ucol = U()->col(d); // local copy
   MatrixXd XX = MatrixXd::Zero(K, K);
   VectorXd yX = VectorXd::Zero(K);

   data()->get_pnm(model(), m_mode, d, yX, XX);

   // add hyperparams
   yX.noalias() += Lambda * mu;
   XX.noalias() += Lambda;

   for(int k=0;k<K;++k) {
       double lambda = XX(k,k);
       double mu = (1/lambda) * (yX(k) - Ucol.transpose() * XX.col(k) + Ucol(k) * XX(k,k));
       double var = randn() / sqrt(lambda);
       Ucol(k) = mu + var;
   }

   U()->col(d) = Ucol;
}

std::ostream &NormalOnePrior::status(std::ostream &os, std::string indent) const
{
   return os;
}
