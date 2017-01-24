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

void NormalPrior::pre_update() {
  tie(mu, Lambda) = CondNormalWishart(U, mu0, b0, WI, df);
}

void NormalPrior::sample_latent(int n)
{

    MatrixXd MM;
    VectorXd rr;

    std::tie(rr, MM) = precision_and_mean(n);

    Eigen::LLT<MatrixXd> chol = MM.llt();
    if(chol.info() != Eigen::Success) {
        throw std::runtime_error("Cholesky Decomposition failed!");
    }

    chol.matrixL().solveInPlace(rr);
    rr.noalias() += nrandn(num_latent());
    chol.matrixU().solveInPlace(rr);
    U.col(n).noalias() = rr;
}

void NormalPrior::savePriorInfo(std::string prefix) {
  writeToCSVfile(prefix + "-" + std::to_string(pos) + "-latentmean.csv", mu);
}


/**
 *  SparseNormalPrior 
 */

SparseNormalPrior::SparseNormalPrior(SparseMF &m, int p, INoiseModel &n)
    : ILatentPrior(m, p, n), SparseLatentPrior(m, p, n), NormalPrior(m, p, n) {}

void SparseNormalPrior::gaussian_precision_and_mean(int n, VectorXd &rr, MatrixXd &MM) 
{
    double alpha = noise.getAlpha();
    for (SparseMatrix<double>::InnerIterator it(Yc, n); it; ++it) {
        auto col = V.col(it.row());
        rr.noalias() += col * (it.value() * alpha);
        MM.triangularView<Eigen::Lower>() += alpha * col * col.transpose();
    }
}

void SparseNormalPrior::probit_precision_and_mean(int n, VectorXd &rr, MatrixXd &MM)
{
    auto u = U.col(n);
    for (SparseMatrix<double>::InnerIterator it(Yc, n); it; ++it) {
        auto col = V.col(it.row());
        MM.triangularView<Eigen::Lower>() += col * col.transpose();
        auto z = (2 * it.value() - 1) * fabs(col.dot(u) + bmrandn_single());
        rr.noalias() += col * z;
    }
}

std::pair<VectorXd,MatrixXd> SparseNormalPrior::precision_and_mean(int n)
{
    const VectorXd &mu_u = getMu(n);
    const MatrixXd &Lambda_u = getLambda(n);

    MatrixXd MM = Lambda_u;
    VectorXd rr = VectorXd::Zero(num_latent());

    if (noise.isProbit()) probit_precision_and_mean(n, rr, MM);
    else gaussian_precision_and_mean(n, rr, MM);

    rr.noalias() += Lambda_u * mu_u;

    return std::make_pair(rr, MM);
}

/**
 *  DenseNormalPrior 
 */


DenseNormalPrior::DenseNormalPrior(DenseMF &m, int p, INoiseModel &n)
    : ILatentPrior(m, p, n), DenseLatentPrior(m, p, n), NormalPrior(m, p, n) {}

std::pair<VectorXd,MatrixXd> DenseNormalPrior::precision_and_mean(int n)
{
    double alpha = noise.getAlpha();
    VectorXd rr = (V * Yc.col(n)) * alpha;
    return std::make_pair(rr, VtV);
}


void DenseNormalPrior::sample_latents() {
    VtV = MatrixNNd::Identity(num_latent(), num_latent());
    double alpha = noise.getAlpha();

#pragma omp parallel for schedule(dynamic, 2) reduction(MatrixPlus:VtV)
    for(int n = 0; n < V.cols(); n++) {
        auto v = V.col(n);
        VtV += v * (alpha * v.transpose());
    }

    DenseLatentPrior::sample_latents();
}

template<class Prior>
MasterNormalPrior<Prior>::MasterNormalPrior(typename Prior::BaseModel &m, int p, INoiseModel &n) 
    : ILatentPrior(m, p, n), Prior(m, p, n), is_init(false) {}


template<typename P1, typename P2>
std::pair<P1, P2> &operator+=(std::pair<P1, P2> &a, const std::pair<P1, P2> &b) {
    a.first += b.first;
    a.second += b.second;
    return a;
}

template<class Prior>
std::pair<Eigen::VectorXd, Eigen::MatrixXd> MasterNormalPrior<Prior>::precision_and_mean(int n) 
{
    // first the master
    auto p = Prior::precision_and_mean(n);

    // then the slaves
    assert(slaves.size() > 0 && "No slaves");
    for(auto &s : slaves) {
        auto &slave_prior = s->priors.at(this->pos);
        p += slave_prior->precision_and_mean(n);
    }

    return p;
}

template<class Prior>
void MasterNormalPrior<Prior>::sample_latents() {
    assert(slaves.size() > 0 && "No slaves");
    if (!is_init) {
        for(auto &s : slaves) s->init();
        is_init = true;
    }

    for(auto &s : slaves) s->step();
    Prior::sample_latents();
}

template<class Prior>
void MasterNormalPrior<Prior>::addSideInfo(Macau &m)
{
    slaves.push_back(&m); 
}

template class MasterNormalPrior<SparseNormalPrior>;
template class MasterNormalPrior<DenseNormalPrior>;

// home made reduction!!
// int count=0;
// int tcount=0;
// #pragma omp threadprivate(tcount)
//  
// omp_set_dynamic(0);
//  
// #pragma omp parallel
// {
// . . .
//  if (event_happened) {
//  tcount++;
//  }
//  . . .
// }
// #pragma omp parallel shared(count)
// {
// #pragma omp atomic
//  count += tcount;
// }

