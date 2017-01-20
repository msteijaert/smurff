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


void SparseNormalPrior::sample_latent(int n)
{
    const VectorXd &mu_u = getMu(n);
    const MatrixXd &Lambda_u = getLambda(n);

    MatrixXd MM = Lambda_u;
    VectorXd rr = VectorXd::Zero(num_latent());

    if (noise.isProbit()) probit_precision_and_mean(n, rr, MM);
    else gaussian_precision_and_mean(n, rr, MM);

    Eigen::LLT<MatrixXd> chol = MM.llt();
    if(chol.info() != Eigen::Success) {
        throw std::runtime_error("Cholesky Decomposition failed!");
    }

    rr.noalias() += Lambda_u * mu_u;
    chol.matrixL().solveInPlace(rr);
    for (int i = 0; i < num_latent(); i++) {
        rr[i] += randn0();
    }
    chol.matrixU().solveInPlace(rr);
    U.col(n).noalias() = rr;
}

void SparseNormalPrior::gaussian_precision_and_mean(int n, VectorXd &rr, MatrixXd &MM) 
{
    double alpha = noise.getAlpha();
    for (SparseMatrix<double>::InnerIterator it(Y, n); it; ++it) {
        auto col = V.col(it.row());
        rr.noalias() += col * ((it.value() - model.mean_rating) * alpha);
        MM.triangularView<Eigen::Lower>() += alpha * col * col.transpose();
    }
}

void SparseNormalPrior::probit_precision_and_mean(int n, VectorXd &rr, MatrixXd &MM)
{
    auto u = U.col(n);
    for (SparseMatrix<double>::InnerIterator it(Y, n); it; ++it) {
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
    auto x = CovF * (V * Y.col(n)) * t + CovL * nrandn(num_latent()).matrix();
    SHOW(x.transpose());
    U.col(n).noalias() = x;
}

void DenseNormalPrior::pre_update()
{
    MatrixNNd VtV = MatrixNNd::Identity(num_latent(), num_latent());
    double t = noise.getAlpha();

#pragma omp parallel for schedule(dynamic, 2) reduction(MatrixPlus:VtV)
    for(int n = 0; n < V.cols(); n++) {
        auto v = V.col(n);
        VtV += v * (t * v.transpose());
    }

    CovF = VtV.inverse();
    CovL = CovF.llt().matrixL();
    CovU = CovF.llt().matrixU();
}

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

