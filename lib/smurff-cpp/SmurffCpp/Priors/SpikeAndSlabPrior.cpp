#include "SpikeAndSlabPrior.h"

#include <IO/MatrixIO.h>

using namespace smurff;
using namespace Eigen;

SpikeAndSlabPrior::SpikeAndSlabPrior(BaseSession &m, int p)
   : ILatentPrior(m, p, "SpikeAndSlabPrior")
{

}

void SpikeAndSlabPrior::init()
{
   const int K = num_latent();
   const int D = num_cols();
   const int nview = data().nview(mode);
   assert(D > 0);

   Zcol.init(MatrixXd::Zero(K,nview));
   W2col.init(MatrixXd::Zero(K,nview));

   //-- prior params
   alpha = ArrayXXd::Ones(K,nview);
   Zkeep = MatrixXd::Constant(K, nview, D);
   r = MatrixXd::Constant(K,nview,.5);
}

void SpikeAndSlabPrior::sample_latents()
{
   ILatentPrior::sample_latents();

   const int nview = data().nview(mode);

   auto Zc = Zcol.combine();
   auto W2c = W2col.combine();

   // update hyper params (per view)
   for(int v=0; v<nview; ++v) {
       const int D = data().view_size(mode, v);

       r.col(v) = ( Zc.col(v).array() + prior_beta ) / ( D + prior_beta * D ) ;
       auto ww = W2c.col(v).array() / 2 + prior_beta_0;
       auto tmpz = Zc.col(v).array() / 2 + prior_alpha_0 ;
       alpha.col(v) = tmpz.binaryExpr(ww, [](double a, double b)->double {
               std::default_random_engine generator;
               std::gamma_distribution<double> distribution(a, 1/b);
               return distribution(generator) + 1e-7;
               });
   }


   Zkeep = Zc.array();
   Zcol.reset();
   W2col.reset();
}

void SpikeAndSlabPrior::sample_latent(int d)
{
   const int K = num_latent();
   const int v = data().view(mode, d);

   auto &W = U(); // alias
   VectorXd Wcol = W.col(d); // local copy

   std::default_random_engine generator;
   std::uniform_real_distribution<double> udist(0,1);
   ArrayXd log_alpha = alpha.col(v).log();
   ArrayXd log_r = - r.col(v).array().log() + (VectorXd::Ones(K) - r.col(v)).array().log();

   MatrixXd XX = MatrixXd::Zero(K, K);
   VectorXd yX = VectorXd::Zero(K);
   data().get_pnm(model(),mode,d,yX,XX);

   for(int k=0;k<K;++k) {
      double lambda = XX(k,k) + alpha(k,v);
      double mu = (1/lambda) * (yX(k) - Wcol.transpose() * XX.col(k) + Wcol(k) * XX(k,k));
      double z1 = log_r(k) -  0.5 * (lambda * mu * mu - log(lambda) + log_alpha(k));
      double z = 1 / (1 + exp(z1));
      double p = udist(generator);
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
   const int V = data().nview(mode);
   for(int v=0; v<V; ++v) {
       int Zcount = (Zkeep.col(v).array() > 0).count();
       os << indent << name << ": Z[" << v << "] = " << Zcount << "/" << num_latent() << "\n";
   }
   return os;
}
