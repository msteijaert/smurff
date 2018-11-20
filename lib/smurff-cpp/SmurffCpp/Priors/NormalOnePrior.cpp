#include "NormalOnePrior.h"
#include <SmurffCpp/IO/MatrixIO.h>

using namespace smurff;

NormalOnePrior::NormalOnePrior(std::shared_ptr<Session> session, uint32_t mode, std::string name)
   : ILatentPrior(session, mode, name)
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
   Lambda *= 10.;

   // parameters of Inv-Whishart distribution
   WI.resize(K, K);
   WI.setIdentity();
   mu0.resize(K);
   mu0.setZero();
   b0 = 2;
   df = K;
}

const Eigen::VectorXd NormalOnePrior::getMu(int n) const
{
   return mu;
}

void NormalOnePrior::update_prior()
{
    std::tie(mu, Lambda) = CondNormalWishart(num_item(), getUUsum(), getUsum(), mu0, b0, WI, df);
}

void NormalOnePrior::sample_latent(int d)
{
   const int K = num_latent();

   Eigen::MatrixXd XX = Eigen::MatrixXd::Zero(K, K);
   Eigen::VectorXd yX = Eigen::VectorXd::Zero(K);

   data().getMuLambda(model(), m_mode, d, yX, XX);

   // add hyperparams
   yX.noalias() += Lambda * mu;
   XX.noalias() += Lambda;

   for(int k=0;k<K;++k) sample_latent(d, k, XX, yX);
}
 
std::pair<double,double> NormalOnePrior::sample_latent(int d, int k, const Eigen::MatrixXd& XX, const Eigen::VectorXd& yX)
{
    auto Ucol = U().col(d);
    double lambda = XX(k,k);
    double mu = (1/lambda) * (yX(k) - Ucol.transpose() * XX.col(k) + Ucol(k) * XX(k,k));
    Ucol(k) = mu + randn() / sqrt(lambda);
    return std::make_pair(mu, lambda);
}

std::ostream &NormalOnePrior::status(std::ostream &os, std::string indent) const
{
   return os;
}
