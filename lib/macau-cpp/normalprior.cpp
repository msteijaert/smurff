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
 *  SparseNormalPrior 
 */

SparseNormalPrior::SparseNormalPrior(SparseMF &m, int p, INoiseModel &n)
    : SparseLatentPrior(m, p, n)
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

void SparseNormalPrior::pre_update() {
  tie(mu, Lambda) = CondNormalWishart(U, mu0, b0, WI, df);
}

void SparseNormalPrior::post_update() {
}

void SparseNormalPrior::savePriorInfo(std::string prefix) {
  writeToCSVfile(prefix + "-" + std::to_string(pos) + "-latentmean.csv", mu);
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


void SparseNormalPrior::sample_latent(int n)
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

/**
 *  DenseNormalPrior 
 */

void DenseNormalPrior::sample_latent(int n)
{
    double t = noise.getAlpha();
   
    MatrixXd MM = VtV;
    VectorXd rr1 = (V * Yc.col(n)) * t;
    VectorXd rr2 = (V * Yc.col(n)) * t;

    Eigen::LLT<MatrixXd> chol = MM.llt();
    if(chol.info() != Eigen::Success) {
        throw std::runtime_error("Cholesky Decomposition failed!");
    }

    //VectorXd random = VectorXd::Zero(num_latent());
    VectorXd random = nrandn(num_latent());
    SHOW(random.transpose());

    chol.matrixU().solveInPlace(rr1);
    SHOW((CovU * rr2).transpose());
    SHOW(rr1.transpose());
    rr1.noalias() += random;
    chol.matrixL().solveInPlace(rr1);
    SHOW(rr1.transpose());

    auto x = CovF * rr2 + CovL * random.matrix();
    SHOW(x.transpose());
    auto x1 = CovL * ((CovU * rr2) + random.matrix());
    SHOW(x1.transpose());

    U.col(n).noalias() = rr1;
}

void DenseNormalPrior::sample_latents() {
    VtV = MatrixNNd::Identity(num_latent(), num_latent());
    double t = noise.getAlpha();

#pragma omp parallel for schedule(dynamic, 2) reduction(MatrixPlus:VtV)
    for(int n = 0; n < V.cols(); n++) {
        auto v = V.col(n);
        VtV += v * (t * v.transpose());
    }

    CovF = VtV.inverse();
    CovL = CovF.llt().matrixL();
    CovU = CovF.llt().matrixU();
    SHOW(VtV);
    SHOW(CovF);
    SHOW(CovU);
    SHOW(CovL);
    SHOW(CovL * CovU);

    DenseLatentPrior::sample_latents();
}

void DenseNormalPrior::pre_update() {}

void DenseNormalPrior::post_update()
{
}


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

