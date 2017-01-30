#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <math.h>
#include <iomanip>

#include "mvnormal.h"
#include "macau.h"
#include "chol.h"
#include "linop.h"
#include "noisemodels.h"
#include "latentprior.h"

extern "C" {
  #include <sparse.h>
}

using namespace std; 
using namespace Eigen;

/**
 *  base class NormalPrior 
 */

NormalPrior::NormalPrior(Factors &m, int p, INoiseModel &n)
    : ILatentPrior(m, p, n) 
{
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

    // FIXME: include siblings!!!!
    tie(mu, Lambda) = CondNormalWishart(U, mu0, b0, WI, df);

    ILatentPrior::sample_latents();
}

void NormalPrior::sample_latent(int n)
{
    const auto &mu_u = getMu(n);
    const auto &Lambda_u = getLambda(n);
    SHOW(Lambda_u);
    const auto &p = pnm(n);
    const double alpha = noise.getAlpha();

    VectorXd rr = p.first * alpha + Lambda_u * mu_u;
    MatrixXd MM = p.second * alpha + Lambda_u;

    Eigen::LLT<MatrixXd> chol = MM.llt();
    if(chol.info() != Eigen::Success) {
        SHOW(V);
        SHOW(p.second);
        SHOW(MM);
        throw std::runtime_error("Cholesky Decomposition failed!");
    }

    chol.matrixL().solveInPlace(rr);
    rr.noalias() += nrandn(num_latent());
    chol.matrixU().solveInPlace(rr);
    SHOW(n);
    SHOW(U.col(n).transpose());
    U.col(n).noalias() = rr;
    SHOW(U.col(n).transpose());
}

void NormalPrior::savePriorInfo(std::string prefix) {
  writeToCSVfile(prefix + "-" + std::to_string(pos) + "-latentmean.csv", mu);
}


/*
 * Master Prior
 */

template<class Prior>
MasterPrior<Prior>::MasterPrior(Factors &m, int p, INoiseModel &n) 
    : Prior(m, p, n), is_init(false) {}
 

template<typename P1, typename P2>
std::pair<P1, P2> &operator+=(std::pair<P1, P2> &a, const std::pair<P1, P2> &b) {
    a.first += b.first;
    a.second += b.second;
    return a;
}

template<class Prior>
std::pair<Eigen::VectorXd, Eigen::MatrixXd> MasterPrior<Prior>::pnm(int n) 
{
    // first the master
    auto p = Prior::pnm(n);

    // then the slaves
    assert(slaves.size() > 0 && "No slaves");
    for(auto &s : slaves) {
        auto &slave_prior = s->priors.at(this->pos);
        p += slave_prior->pnm(n);
    }

    return p;
}

template<class Prior>
void MasterPrior<Prior>::sample_latents() {
    assert(slaves.size() > 0 && "No slaves");
    if (!is_init) {
        for(auto &s : slaves) s->init();
        is_init = true;
    }

    //for(auto &s : slaves) s->step();

    Prior::sample_latents();
}

//template<class Prior>
//void MasterPrior<Prior>::addSideInfo(Factors &info, Macau &macau)
//{
//    slaves.push_back(Macau(macau, info)); 
//}

template class MasterPrior<NormalPrior>;
template class MasterPrior<SpikeAndSlabPrior>;

