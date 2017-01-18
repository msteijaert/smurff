#include "noisemodels.h"
#include "latentprior.h"

template<class BasePrior, class BaseModel>
SpikeAndSlabPrior<BasePrior,BaseModel>::SpikeAndSlabPrior(BaseModel &m, int p, INoiseModel &n)
    : BasePrior(m, p, n)
{
    const int K = this->num_latent();
    const int D = this->U.cols();
    assert(D > 0);
    
    //-- prior params
    alpha = ArrayNd::Ones(K);
    Zcol = VectorNd::Zero(K);
    Zkeep = VectorNd::Constant(K, D);
    W2col = VectorNd::Zero(K);
    r = VectorNd::Constant(K,.5);
}

template<class BasePrior, class BaseModel>
void SpikeAndSlabPrior<BasePrior,BaseModel>::sample_latent(int d)
{
    const int K = this->num_latent();
    auto &W = this->U; // aliases
    auto &X = this->V; // aliases
    
    std::default_random_engine generator;
    std::uniform_real_distribution<double> udist(0,1);
    ArrayNd log_alpha = alpha.log();
    ArrayNd log_r = - r.array().log() + (VectorNd::Ones(K) - r).array().log();

    MatrixNNd XX(MatrixNNd::Zero(K,K));
    VectorNd Wcol = W.col(d);
    VectorNd yX(VectorNd::Zero(K));
    //for (SparseMatrixD::InnerIterator it(this->Y,d); it; ++it) {
    for(auto it = this->model.it(d); it; ++it) {
        double y = it.value() - this->model.mean_rating;
        auto Xcol = X.col(it.row());
        yX.noalias() += y * Xcol;
        XX.noalias() += Xcol * Xcol.transpose();
    }

    double t = this->noise.sample(d, 0).first;

    for(unsigned k=0;k<K;++k) {
        double lambda = t * XX(k,k) + alpha(k);
        double mu = t / lambda * (yX(k) - Wcol.transpose() * XX.col(k) + Wcol(k) * XX(k,k));
        double z1 = log_r(k) -  0.5 * (lambda * mu * mu - log(lambda) + log_alpha(k));
        double z = 1 / (1 + exp(z1));
        double p = udist(generator);
        if (Zkeep(k) > 0 && p < z) {
            Zcol(k)++;
            Wcol(k) = mu + randn() / sqrt(lambda);
        } else {
            Wcol(k) = .0;
        }
    }

    W.col(d) = Wcol;
    W2col += Wcol.array().square().matrix();
}

template<class BasePrior, class BaseModel>
void SpikeAndSlabPrior<BasePrior,BaseModel>::savePriorInfo(std::string prefix) {
}


template<class BasePrior, class BaseModel>
void SpikeAndSlabPrior<BasePrior,BaseModel>::pre_update() {
}

template<class BasePrior, class BaseModel>
void SpikeAndSlabPrior<BasePrior,BaseModel>::post_update() {
    const int D = this->U.cols();
    
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
}

template class SpikeAndSlabPrior<SparseLatentPrior, SparseMF>;
template class SpikeAndSlabPrior<DenseLatentPrior, DenseMF>;
