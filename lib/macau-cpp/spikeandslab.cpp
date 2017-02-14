#include "noisemodels.h"
#include "macau.h"

namespace Macau {

SpikeAndSlabPrior::SpikeAndSlabPrior(BaseSession &m, int p)
    : ILatentPrior(m, p, "SpikeAndSlabPrior") {}

void SpikeAndSlabPrior::init() {
    const int K = num_latent();
    const int D = U.cols();
    assert(D > 0);
    
    //-- prior params
    alpha = ArrayNd::Ones(K);
    Zcol = VectorNd::Zero(K);
    Zkeep = VectorNd::Constant(K, D);
    W2col = VectorNd::Zero(K);
    r = VectorNd::Constant(K,.5);
}


void SpikeAndSlabPrior::addSibling(BaseSession &b) 
{
     addSiblingTempl<SpikeAndSlabPrior>(b);
}


void SpikeAndSlabPrior::sample_latent(int d)
{
    const int K = num_latent();
    auto &W = U; // aliases
    VectorNd Wcol = W.col(d); // local copy
    
    std::default_random_engine generator;
    std::uniform_real_distribution<double> udist(0,1);
    ArrayNd log_alpha = alpha.log();
    ArrayNd log_r = - r.array().log() + (VectorNd::Ones(K) - r).array().log();

    MatrixNNd XX = MatrixNNd::Zero(num_latent(), num_latent());
    VectorNd yX = VectorNd::Zero(num_latent());
    pnm(d, yX, XX);
    double t = noise().getAlpha();

    for(int k=0;k<K;++k) {
        double lambda = t * XX(k,k) + alpha(k);
        double mu = t / lambda * (yX(k) - Wcol.transpose() * XX.col(k) + Wcol(k) * XX(k,k));
        double z1 = log_r(k) -  0.5 * (lambda * mu * mu - log(lambda) + log_alpha(k));
        double z = 1 / (1 + exp(z1));
        double p = udist(generator);
        if (Zkeep(k) > 0 && p < z) {
            Zcol(k)++;
            double var = randn() / sqrt(lambda);
            Wcol(k) = mu + var;
            assert(mu < 100.);
        } else {
            Wcol(k) = .0;
        }
    }
    W.col(d) = Wcol;
    W2col += Wcol.array().square().matrix();
}

void SpikeAndSlabPrior::sample_latents() {
    ILatentPrior::sample_latents();

    const int D = U.cols();
 
    // one: accumulate across siblings
    for(auto s : siblings) {
        auto p = dynamic_cast<SpikeAndSlabPrior *>(s);
        Zcol += p->Zcol;
        W2col += p->W2col;
    }
    
    // two: update hyper params
    r = ( Zcol.array() + prior_beta ) / ( D + prior_beta * D ) ;

    //-- updata alpha K samples from Gamma
    auto ww = W2col.array() / 2 + prior_beta_0;
    auto tmpz = Zcol.array() / 2 + prior_alpha_0 ;
    alpha = tmpz.binaryExpr(ww, [](double a, double b)->double {
            std::default_random_engine generator;
            std::gamma_distribution<double> distribution(a, 1/b);
            return distribution(generator) + 1e-7;
    });

    Zkeep = Zcol.array();
    Zcol.setZero();
    W2col.setZero();

    // three: update siblings
    for(auto s : siblings) {
        auto p = dynamic_cast<SpikeAndSlabPrior *>(s);
        p->Zcol.setZero();
        p->W2col.setZero();
        p->Zkeep = Zkeep;
        p->alpha = alpha;
        p->r = r;
    }
}

} // end namespace Macau
