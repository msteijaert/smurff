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

ILatentPrior::ILatentPrior(MFactor &d) : fac(d) {}

void ILatentPrior::sample_latents(const Eigen::MatrixXd &V) {
#pragma omp parallel for schedule(dynamic, 2)
  for(int n = 0; n < fac.U.cols(); n++) sample_latent(n, V); 
}


/**
 *  NormalPrior 
 */

template<class NoiseModel>
NormalPrior<NoiseModel>::NormalPrior(MFactor &f, NoiseModel &noise)
    : ILatentPrior(f), noise(noise)
{
  mu.resize(num_latent());
  mu.setZero();

  Lambda.resize(num_latent(), num_latent());
  Lambda.setIdentity();
  Lambda *= 10;

  // parameters of Inv-Whishart distribution
  WI.resize(num_latent(), num_latent());
  WI.setIdentity();
  mu0.resize(num_latent());
  mu0.setZero();
  b0 = 2;
  df = num_latent();
}

template<class NoiseModel>
void NormalPrior<NoiseModel>::sample_latent(int n, const MatrixXd &V)
{
  const VectorXd &mu_u = getMu(n);
  const MatrixXd &Lambda_u = getLambda(n);
  auto &Y = fac.Y;
  auto &U = fac.U;
  double mean_rating = fac.mean_rating;

  MatrixXd MM = Lambda_u;
  VectorXd rr = VectorXd::Zero(num_latent());
  for (SparseMatrix<double>::InnerIterator it(Y, n); it; ++it) {
    auto col = U.col(it.row());
    std::pair<double, double> alpha = noise.sample(n, it.row());
    rr.noalias() += col * ((it.value() - mean_rating) * alpha.first);
    MM.triangularView<Eigen::Lower>() += alpha.second * col * col.transpose();
  }

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

template<class NoiseModel>
void NormalPrior<NoiseModel>::update_prior() {
  tie(mu, Lambda) = CondNormalWishart(fac.U, mu0, b0, WI, df);
}


template<class NoiseModel>
void NormalPrior<NoiseModel>::saveModel(std::string prefix) {
  writeToCSVfile(prefix + "-latentmean.csv", mu);
}

/** MacauPrior */
template<class FType, class NoiseModel>
MacauPrior<FType, NoiseModel>::MacauPrior(MFactor &data, NoiseModel &noise, std::unique_ptr<FType> &Fmat, bool comp_FtF)
    : NormalPrior<NoiseModel>(data, noise)
{
  auto U = this->fac.U;

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

template<class FType, class NoiseModel>
void MacauPrior<FType, NoiseModel>::update_prior() {
  // residual (Uhat is later overwritten):
  Uhat.noalias() = this->fac.U - Uhat;
  MatrixXd BBt = A_mul_At_combo(beta);
  // sampling Gaussian
  tie(this->mu, this->Lambda) = CondNormalWishart(Uhat, this->mu0, this->b0, this->WI + lambda_beta * BBt, this->df + beta.cols());
  sample_beta();
  compute_uhat(Uhat, *F, beta);
  lambda_beta = sample_lambda_beta(beta, this->Lambda, lambda_beta_nu0, lambda_beta_mu0);
}

template<class FType, class NoiseModel>
double MacauPrior<FType, NoiseModel>::getLinkNorm() {
  return beta.norm();
}

/** Update beta and Uhat */
template<class FType, class NoiseModel>
void MacauPrior<FType, NoiseModel>::sample_beta() {
  const int num_feat = beta.cols();
  // Ft_y = (U .- mu + Normal(0, Lambda^-1)) * F + sqrt(lambda_beta) * Normal(0, Lambda^-1)
  // Ft_y is [ D x F ] matrix
  MatrixXd tmp = (this->fac.U + MvNormal_prec_omp(this->Lambda, this->fac.U.cols())).colwise() - this->mu;
  MatrixXd Ft_y = A_mul_B(tmp, *F) + sqrt(lambda_beta) * MvNormal_prec_omp(this->Lambda, num_feat);

  if (use_FtF) {
    MatrixXd K(FtF.rows(), FtF.cols());
    K.triangularView<Eigen::Lower>() = FtF;
    K.diagonal().array() += lambda_beta;
    chol_decomp(K);
    chol_solve_t(K, Ft_y);
    beta = Ft_y;
  } else {
    // BlockCG
    solve_blockcg(beta, *F, lambda_beta, Ft_y, tol, 32, 8);
  }
}

template<class FType, class NoiseModel>
void MacauPrior<FType, NoiseModel>::saveModel(std::string prefix) {
  writeToCSVfile(prefix + "-latentmean.csv", this->mu);
  writeToCSVfile(prefix + "-link.csv", this->beta);
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

template class MacauPrior<SparseFeat, FixedGaussianNoise>;
template class MacauPrior<SparseFeat, AdaptiveGaussianNoise>;
template class MacauPrior<SparseFeat, ProbitNoise>;
template class MacauPrior<SparseDoubleFeat, FixedGaussianNoise>;
template class MacauPrior<SparseDoubleFeat, AdaptiveGaussianNoise>;
template class MacauPrior<SparseDoubleFeat, ProbitNoise>;
template class NormalPrior<FixedGaussianNoise>;
template class NormalPrior<AdaptiveGaussianNoise>;
template class NormalPrior<ProbitNoise>;

//template class MacauPrior<Eigen::MatrixXd>;

