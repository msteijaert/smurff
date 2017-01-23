#include "noisemodels.h"
#include "latentprior.h"

SpikeAndSlabPrior::SpikeAndSlabPrior(Factors &m, int p, INoiseModel &n)
    : ILatentPrior(m, p, n)
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

void SpikeAndSlabPrior::sample_latent(int d)
{
    const int K = this->num_latent();
    auto &W = this->U; // aliases
    VectorNd Wcol = W.col(d); // local copy
    
    std::default_random_engine generator;
    std::uniform_real_distribution<double> udist(0,1);
    ArrayNd log_alpha = alpha.log();
    ArrayNd log_r = - r.array().log() + (VectorNd::Ones(K) - r).array().log();

    MatrixNNd XX(MatrixNNd::Zero(K,K));
    VectorNd yX(VectorNd::Zero(K));
    compute_XX_yX(d, XX, yX);
    double t = this->noise.getAlpha();

    for(unsigned k=0;k<K;++k) {
        double lambda = t * XX(k,k) + alpha(k);
        SHOW(lambda);
        SHOW(alpha(k));
        double mu = t / lambda * (yX(k) - Wcol.transpose() * XX.col(k) + Wcol(k) * XX(k,k));
        double z1 = log_r(k) -  0.5 * (lambda * mu * mu - log(lambda) + log_alpha(k));
        double z = 1 / (1 + exp(z1));
        double p = udist(generator);
        if (Zkeep(k) > 0 && p < z) {
            Zcol(k)++;
            double var = randn() / sqrt(lambda);
            SHOW(var);
            Wcol(k) = mu + var;
        } else {
            Wcol(k) = .0;
        }
    }
    SHOW(Wcol.transpose());
    auto &X = this->V; // aliases
    SHOW(Wcol.transpose() * X);

    W.col(d) = Wcol;
    W2col += Wcol.array().square().matrix();
}

SparseSpikeAndSlabPrior::SparseSpikeAndSlabPrior(SparseMF &m, int p, INoiseModel &n)
    : ILatentPrior(m, p, n), SpikeAndSlabPrior(m, p, n), SparseLatentPrior(m, p, n) {}

void SparseSpikeAndSlabPrior::compute_XX_yX(int d, Eigen::MatrixXd &XX, Eigen::VectorXd &yX)
{
    auto &X = this->V; // aliases
    for (SparseMatrixD::InnerIterator it(this->Yc,d); it; ++it) {
        double y = it.value();
        auto Xcol = X.col(it.row());
        yX.noalias() += y * Xcol;
        XX.noalias() += Xcol * Xcol.transpose();
    }
}

DenseSpikeAndSlabPrior::DenseSpikeAndSlabPrior(DenseMF &m, int p, INoiseModel &n)
    : ILatentPrior(m, p, n), SpikeAndSlabPrior(m, p, n), DenseLatentPrior(m, p, n) {}

void DenseSpikeAndSlabPrior::compute_XX_yX(int d, Eigen::MatrixXd &XX, Eigen::VectorXd &yX)
{
    auto &X = this->V; // aliases
    XX = this->XX;
    SHOW(XX);
    SHOW(X * X.transpose());
   // assert(XX == X * X.transpose());

    yX = Yc.col(d).transpose() * X.transpose();
}


void SpikeAndSlabPrior::savePriorInfo(std::string prefix) { }


void DenseSpikeAndSlabPrior::pre_update() {
    XX.setZero();

#pragma omp parallel for schedule(dynamic, 2) reduction(MatrixPlus:XX)
    for(int n = 0; n < V.cols(); n++) {
        auto v = V.col(n);
        XX += v * v.transpose();
    }

    if (!this->is_init) {
        const int K = this->num_latent();
        double t = noise.getAlpha();
        auto &X = this->V; // aliases

        MatrixNNd covW = (MatrixNNd::Identity(K, K) + t * XX).inverse();
        MatrixNNd Sx = covW.llt().matrixU();
        this->U = covW * (X * (Yc.array() - model.mean_rating).matrix()) * t + Sx * nrandn(K,U.cols()).matrix();
        this->is_init = true;
    }
}


void SpikeAndSlabPrior::post_update() {
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
    W2col.setZero();
}

