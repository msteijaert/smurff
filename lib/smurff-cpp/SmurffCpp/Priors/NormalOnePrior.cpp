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
}

void NormalOnePrior::update_prior()
{
}

void NormalOnePrior::sample_latent(int d)
{
   const int K = num_latent();

   VectorXd Wcol = U()->col(d); // local copy
   MatrixXd XX = MatrixXd::Zero(K, K);
   VectorXd yX = VectorXd::Zero(K);
   data()->get_pnm(model(), m_mode, d, yX, XX);

   for(int k=0;k<K;++k) {
       double lambda = XX(k,k);
       double mu = (1/lambda) * (yX(k) - Wcol.transpose() * XX.col(k) + Wcol(k) * XX(k,k));
       double var = randn() / sqrt(lambda);
       Wcol(k) = mu + var;
   }

   U()->col(d) = Wcol;
}

std::ostream &NormalOnePrior::status(std::ostream &os, std::string indent) const
{
   return os;
}
