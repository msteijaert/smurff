#include "SpikeAndSlabPrior.h"
#include <SmurffCpp/IO/MatrixIO.h>

using namespace smurff;
using namespace Eigen;

SpikeAndSlabPrior::SpikeAndSlabPrior(std::shared_ptr<BaseSession> session, int mode)
   : ILatentPrior(session, mode, "SpikeAndSlabPrior")
{

}

void SpikeAndSlabPrior::init()
{
   const int K = num_latent();
   const int D = num_cols();
   const int nview = data().nview(m_mode);
   assert(D > 0);

   Zcol.init(MatrixXd::Zero(K,nview));
   W2col.init(MatrixXd::Zero(K,nview));

   //-- prior params
   alpha = ArrayXXd::Ones(K,nview);
   Zkeep = MatrixXd::Constant(K, nview, D);
   r = MatrixXd::Constant(K,nview,.5);
}

void SpikeAndSlabPrior::update_prior()
{
   const int nview = data().nview(m_mode);
   
   auto Zc = Zcol.combine();
   auto W2c = W2col.combine();

   // update hyper params (per view)
   for(int v=0; v<nview; ++v) {
       const int D = data().view_size(m_mode, v);
       r.col(v) = ( Zc.col(v).array() + prior_beta ) / ( D + prior_beta * D ) ;
       auto ww = W2c.col(v).array() / 2 + prior_beta_0;
       auto tmpz = Zc.col(v).array() / 2 + prior_alpha_0 ;
       alpha.col(v) = tmpz.binaryExpr(ww, [](double a, double b)->double {
               return rgamma(a, 1/b) + 1e-7;
       });
   }


   Zkeep = Zc.array();
   Zcol.reset();
   W2col.reset();
}

void SpikeAndSlabPrior::sample_latent(int d)
{
   const int K = num_latent();
   const int v = data().view(m_mode, d);

   auto &W = U(); // alias
   VectorXd Wcol = W.col(d); // local copy

   ArrayXd log_alpha = alpha.col(v).log();
   ArrayXd log_r = - r.col(v).array().log() + (VectorXd::Ones(K) - r.col(v)).array().log();

   MatrixXd XX = MatrixXd::Zero(K, K);
   VectorXd yX = VectorXd::Zero(K);
   data().get_pnm(model(), m_mode, d, yX, XX);

   for(int k=0;k<K;++k) {
      double lambda = XX(k,k) + alpha(k,v);
      double mu = (1/lambda) * (yX(k) - Wcol.transpose() * XX.col(k) + Wcol(k) * XX(k,k));
      double z1 = log_r(k) -  0.5 * (lambda * mu * mu - log(lambda) + log_alpha(k));
      double z = 1 / (1 + exp(z1));
      double p = rand_unif(0,1);
      if (Zkeep(k,v) > 0 && p < z) {
         Zcol.local()(k,v)++;
         double var = randn() / sqrt(lambda);
         Wcol(k) = mu + var;
      } else {
         Wcol(k) = .0;
      }
   }

   W.col(d) = Wcol;
   W2col.local().col(v) += Wcol.array().square().matrix();
}

std::ostream &SpikeAndSlabPrior::status(std::ostream &os, std::string indent) const
{
   const int V = data().nview(m_mode);
   for(int v=0; v<V; ++v) 
   {
       int Zcount = (Zkeep.col(v).array() > 0).count();
       os << indent << m_name << ": Z[" << v << "] = " << Zcount << "/" << num_latent() << "\n";
   }
   return os;
}
