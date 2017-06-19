#include "noisemodels.h"
#include "session.h"
#include "data.h"

using namespace Eigen;

namespace smurff {

SpikeAndSlabPrior::SpikeAndSlabPrior(BaseSession &m, int p)
    : ILatentPrior(m, p, "SpikeAndSlabPrior") {}


void SpikeAndSlabPrior::init() {
    const int K = num_latent();
    const int D = num_cols();
    assert(D > 0);

    Zcol.init(VectorXd::Zero(num_latent()));
    W2col.init(VectorXd::Zero(num_latent()));
    
    //-- prior params
    alpha = ArrayNd::Ones(K);
    Zkeep = VectorNd::Constant(K, D);
    r = VectorNd::Constant(K,.5);
}

void SpikeAndSlabPrior::sample_latent(int d)
{
    const int K = num_latent();

    auto &W = U(); // aliases
    VectorNd Wcol = W.col(d); // local copy
    
    std::default_random_engine generator;
    std::uniform_real_distribution<double> udist(0,1);
    ArrayNd log_alpha = alpha.log();
    ArrayNd log_r = - r.array().log() + (VectorNd::Ones(K) - r).array().log();

    MatrixNNd XX = MatrixNNd::Zero(num_latent(), num_latent());
    VectorNd yX = VectorNd::Zero(num_latent());
    session.data->get_pnm(model(),mode,d,yX,XX);
    double t = noise().getAlpha();

    for(int k=0;k<K;++k) {
        double lambda = t * XX(k,k) + alpha(k);
        double mu = t / lambda * (yX(k) - Wcol.transpose() * XX.col(k) + Wcol(k) * XX(k,k));
        double z1 = log_r(k) -  0.5 * (lambda * mu * mu - log(lambda) + log_alpha(k));
        double z = 1 / (1 + exp(z1));
        double p = udist(generator);
        if (Zkeep(k) > 0 && p < z) {
            Zcol.local()(k)++;
            double var = randn() / sqrt(lambda);
            Wcol(k) = mu + var;
        } else {
            Wcol(k) = .0;
        }
    }

    W.col(d) = Wcol;
    W2col.local() += Wcol.array().square().matrix();
}

void SpikeAndSlabPrior::sample_latents() {
    ILatentPrior::sample_latents();

    const int D = num_cols();
    auto Zc = Zcol.combine();
    auto W2c = W2col.combine();
 
    // update hyper params
    r = ( Zc.array() + prior_beta ) / ( D + prior_beta * D ) ;
    auto ww = W2c.array() / 2 + prior_beta_0;
    auto tmpz = Zc.array() / 2 + prior_alpha_0 ;
    alpha = tmpz.binaryExpr(ww, [](double a, double b)->double {
            std::default_random_engine generator;
            std::gamma_distribution<double> distribution(a, 1/b);
            return distribution(generator) + 1e-7;
    });


    Zkeep = Zc.array();
    Zcol.reset();
    W2col.reset();
}

std::string SpikeAndSlabPrior::status() const {
    int Zcount = (Zkeep.array() > 0).count();
    return name + " Z = " + std::to_string(Zcount) + "/" + std::to_string(num_latent());
}

} // end namespace smurff
