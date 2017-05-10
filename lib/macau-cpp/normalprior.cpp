#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <iomanip>

#include "mvnormal.h"
#include "session.h"
#include "chol.h"
#include "linop.h"
#include "noisemodels.h"
#include "latentprior.h"

using namespace std; 
using namespace Eigen;

namespace Macau {

ILatentPrior::ILatentPrior(BaseSession &m, int p, std::string name)
    : session(m), mode(p), name(name) {} 

std::ostream &ILatentPrior::info(std::ostream &os, std::string indent) 
{
    os << indent << mode << ": " << name << "\n";
    return os;
}

Model &ILatentPrior::model() { return session.model; }
Data &ILatentPrior::data() { return *session.data; }
INoiseModel &ILatentPrior::noise() { return *data().noise; }
MatrixXd &ILatentPrior::U() { return model().U(mode); }
MatrixXd &ILatentPrior::V() { return model().V(mode); }

void ILatentPrior::init() 
{
    rrs.init(VectorNd::Zero(num_latent()));
    MMs.init(MatrixNNd::Zero(num_latent(), num_latent()));
}

void ILatentPrior::sample_latents() 
{
    session.data->update_pnm(model(), mode);
#pragma omp parallel for schedule(guided)
    for(int n = 0; n < U().cols(); n++) {
#pragma omp task
        sample_latent(n); 
    }
}

/**
 *  base class NormalPrior 
 */

NormalPrior::NormalPrior(BaseSession &m, int p, std::string name)
    : ILatentPrior(m, p, name) {}

void NormalPrior::init() {
    ILatentPrior::init();

    Ucol.init(VectorNd::Zero(num_latent())),
    UUcol.init(MatrixNNd::Zero(num_latent(), num_latent()));

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

void NormalPrior::sample_latents() {

    const int N = num_cols();
    const auto cov = UUcol.combine();
    const auto sum = Ucol.combine();
    tie(mu, Lambda) = CondNormalWishart(N, cov / N, sum / N, mu0, b0, WI, df);

    UUcol.reset();
    Ucol.reset();

    ILatentPrior::sample_latents();
}

void NormalPrior::sample_latent(int n)
{
    const auto &mu_u = getMu(n);

    VectorNd &rr = rrs.local();
    MatrixNNd &MM = MMs.local();

    rr.setZero();
    MM.setZero();

    // add pnm
    session.data->get_pnm(model(),mode,n,rr,MM);

    // add hyperparams
    rr.noalias() += Lambda * mu_u;
    MM.noalias() += Lambda;

    Eigen::LLT<MatrixXd> chol = MM.llt();
    if(chol.info() != Eigen::Success) {
        throw std::runtime_error("Cholesky Decomposition failed!");
    }

    chol.matrixL().solveInPlace(rr);
    rr.noalias() += nrandn(num_latent());
    chol.matrixU().solveInPlace(rr);

    U().col(n).noalias() = rr;
    Ucol.local().noalias() += rr;
    UUcol.local().noalias() += rr * rr.transpose();

}

void NormalPrior::save(std::string prefix, std::string suffix) {
  write_dense(prefix + "-U" + std::to_string(mode) + "-latentmean" + suffix, mu);
}

void NormalPrior::restore(std::string prefix, std::string suffix) {
  read_dense(prefix + "-U" + std::to_string(mode) + "-latentmean" + suffix, mu);
}

} // end namespace Macau
