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
    std::tie(rr, MM) = pnm(n);

    double alpha = noise.getAlpha();
    rr *= alpha;
    MM *= alpha;

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



/**
 *  DenseNormalPrior 
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

    for(auto &s : slaves) s->step();
    Prior::sample_latents();
}

template<class Prior>
void MasterPrior<Prior>::addSideInfo(Macau &m)
{
    slaves.push_back(&m); 
}

template class MasterPrior<NormalPrior>;
template class MasterPrior<SpikeAndSlabPrior>;

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

