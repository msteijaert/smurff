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
#include "macauprior.h"

extern "C" {
  #include <sparse.h>
}

using namespace std; 
using namespace Eigen;

Eigen::MatrixXd A_mul_B(Eigen::MatrixXd & A, Eigen::MatrixXd & B);
Eigen::MatrixXd A_mul_B(Eigen::MatrixXd & A, SparseFeat & B);
Eigen::MatrixXd A_mul_B(Eigen::MatrixXd & A, SparseDoubleFeat & B);

template<class FType>
MacauPrior<FType>::MacauPrior(MacauBase &m, int p)
    : NormalPrior(m, p)  {}

template<class FType>
void MacauPrior<FType>::addSibling(MacauBase &b) 
{
     addSiblingTempl<MacauPrior<FType>>(b);
}
    
template<class FType>
void MacauPrior<FType>::addSideInfo(std::unique_ptr<FType> &Fmat, bool comp_FtF)
{
    assert((Fmat->rows() == U.rows()) && "Number of rows in train must be equal to number of rows in features");

    // side information
    F = std::move(Fmat);
    use_FtF = comp_FtF;
    if (use_FtF) {
        FtF.resize(F->cols(), F->cols());
        At_mul_A(FtF, *F);
    }

    Uhat.resize(this->num_latent(), F->rows());
    Uhat.setZero();

    beta.resize(this->num_latent(), F->cols());
    beta.setZero();

    // initial value (should be determined automatically)
    lambda_beta = 5.0;
    // Hyper-prior for lambda_beta (mean 1.0, var of 1e+3):
    lambda_beta_mu0 = 1.0;
    lambda_beta_nu0 = 1e-3;
}

template<class FType>
void MacauPrior<FType>::sample_latents() {
  NormalPrior::sample_latents();

  // residual (Uhat is later overwritten):
  Uhat.noalias() = U - Uhat;
  MatrixXd BBt = A_mul_At_combo(beta);
  // sampling Gaussian
  tie(this->mu, this->Lambda) = CondNormalWishart(Uhat, this->mu0, this->b0, this->WI + lambda_beta * BBt, this->df + beta.cols());
  sample_beta();
  compute_uhat(Uhat, *F, beta);
  lambda_beta = sample_lambda_beta(beta, this->Lambda, lambda_beta_nu0, lambda_beta_mu0);
}

template<class FType>
double MacauPrior<FType>::getLinkNorm() {
  return beta.norm();
}

template<class FType>
void MacauPrior<FType>::compute_Ft_y_omp(MatrixXd &Ft_y) {
    const int num_feat = beta.cols();

    // Ft_y = (U .- mu + Normal(0, Lambda^-1)) * F + sqrt(lambda_beta) * Normal(0, Lambda^-1)
    // Ft_y is [ D x F ] matrix
    MatrixXd tmp = (U + MvNormal_prec_omp(Lambda, U.cols())).colwise() - mu;
    Ft_y = A_mul_B(tmp, *F);
    MatrixXd tmp2 = MvNormal_prec_omp(Lambda, num_feat);

#pragma omp parallel for schedule(static)
    for (int f = 0; f < num_feat; f++) {
        for (int d = 0; d < num_latent(); d++) {
            Ft_y(d, f) += sqrt(lambda_beta) * tmp2(d, f);
        }
    }
}

/** Update beta and Uhat */
template<class FType>
void MacauPrior<FType>::sample_beta() {
    MatrixXd Ft_y;
    this->compute_Ft_y_omp(Ft_y);

    if (use_FtF) {
        // direct method
        MatrixXd K(FtF.rows(), FtF.cols());
        K.triangularView<Eigen::Lower>() = FtF;
        K.diagonal().array() += lambda_beta;
        chol_decomp(K);
        chol_solve_t(K, Ft_y);
        beta = Ft_y;
    } else {
        // BlockCG solver
        solve_blockcg(beta, *F, lambda_beta, Ft_y, tol, 32, 8);
    }
}

template<class FType>
void MacauPrior<FType>::savePriorInfo(std::string prefix) {
  writeToCSVfile(prefix + "-" + std::to_string(pos) + "-latentmean.csv", this->mu);
  writeToCSVfile(prefix + "-" + std::to_string(pos) + "-link.csv", this->beta);
}

std::pair<double,double> posterior_lambda_beta(Eigen::MatrixXd & beta, Eigen::MatrixXd & Lambda_u, double nu, double mu) {
  const int D = beta.rows();
  MatrixXd BB(D, D);
  A_mul_At_combo(BB, beta);
  double nux = nu + beta.rows() * beta.cols();
  double mux = mu * nux / (nu + mu * (BB.selfadjointView<Eigen::Lower>() * Lambda_u).trace() );
  double b   = nux / 2;
  double c   = 2 * mux / nux;
  return std::make_pair(b, c);
}

double sample_lambda_beta(Eigen::MatrixXd & beta, Eigen::MatrixXd & Lambda_u, double nu, double mu) {
  auto gamma_post = posterior_lambda_beta(beta, Lambda_u, nu, mu);
  return rgamma(gamma_post.first, gamma_post.second);
}

/**
 *
 * X = A * B
 */
Eigen::MatrixXd A_mul_B(Eigen::MatrixXd & A, Eigen::MatrixXd & B) {
  MatrixXd out(A.rows(), B.cols());
  A_mul_B_blas(out, A, B);
  return out;
}

Eigen::MatrixXd A_mul_B(Eigen::MatrixXd & A, SparseFeat & B) {
  MatrixXd out(A.rows(), B.cols());
  A_mul_Bt(out, B.Mt, A);
  return out;
}

Eigen::MatrixXd A_mul_B(Eigen::MatrixXd & A, SparseDoubleFeat & B) {
  MatrixXd out(A.rows(), B.cols());
  A_mul_Bt(out, B.Mt, A);
  return out;
}

template class MacauPrior<SparseFeat>;
template class MacauPrior<SparseDoubleFeat>;
//template class MacauPrior<Eigen::MatrixXd>;


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

