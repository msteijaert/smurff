#include "MacauPrior.h"

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <math.h>
#include <iomanip>

#include "mvnormal.h"
#include "session.h"
#include "chol.h"
#include "linop.h"

#include "data.h"

extern "C" {
  #include <sparse.h>
}

using namespace Eigen;
using namespace smurff;

//X = A * B
Eigen::MatrixXd A_mul_B(Eigen::MatrixXd & A, Eigen::MatrixXd & B) 
{
   MatrixXd out(A.rows(), B.cols());
   A_mul_B_blas(out, A, B);
   return out;
}
 
Eigen::MatrixXd A_mul_B(Eigen::MatrixXd & A, SparseFeat & B) 
{
   MatrixXd out(A.rows(), B.cols());
   A_mul_Bt(out, B.Mt, A);
   return out;
}
 
Eigen::MatrixXd A_mul_B(Eigen::MatrixXd & A, SparseDoubleFeat & B) 
{
   MatrixXd out(A.rows(), B.cols());
   A_mul_Bt(out, B.Mt, A);
   return out;
}

namespace smurff{

std::pair<double,double> posterior_lambda_beta(Eigen::MatrixXd & beta, Eigen::MatrixXd & Lambda_u, double nu, double mu) 
{
   const int D = beta.rows();
   MatrixXd BB(D, D);
   A_mul_At_combo(BB, beta);
   double nux = nu + beta.rows() * beta.cols();
   double mux = mu * nux / (nu + mu * (BB.selfadjointView<Eigen::Lower>() * Lambda_u).trace() );
   double b   = nux / 2;
   double c   = 2 * mux / nux;
   return std::make_pair(b, c);
}
 
double sample_lambda_beta(Eigen::MatrixXd & beta, Eigen::MatrixXd & Lambda_u, double nu, double mu) 
{
   auto gamma_post = smurff::posterior_lambda_beta(beta, Lambda_u, nu, mu);
   return rgamma(gamma_post.first, gamma_post.second);
}

}

template<class FType>
MacauPrior<FType>::MacauPrior(BaseSession &m, int p)
   : NormalPrior(m, p, "MacauPrior")  
{
   
}

template<class FType>
void MacauPrior<FType>::init()
{
   NormalPrior::init();

   assert((F->rows() == num_cols()) && 
         "Number of rows in train must be equal to number of rows in features");

   if (use_FtF) 
   {
      FtF.resize(F->cols(), F->cols());
      At_mul_A(FtF, *F);
   }

   Uhat.resize(this->num_latent(), F->rows());
   Uhat.setZero();

   beta.resize(this->num_latent(), F->cols());
   beta.setZero();
}

template<class FType>
void MacauPrior<FType>::sample_latents() {
  NormalPrior::sample_latents();

  // residual (Uhat is later overwritten):
  Uhat.noalias() = U() - Uhat;
  MatrixXd BBt = A_mul_At_combo(beta);
  // sampling Gaussian
  std::tie(this->mu, this->Lambda) = CondNormalWishart(Uhat, this->mu0, this->b0, this->WI + lambda_beta * BBt, this->df + beta.cols());
  sample_beta();
  compute_uhat(Uhat, *F, beta);
  lambda_beta = smurff::sample_lambda_beta(beta, this->Lambda, lambda_beta_nu0, lambda_beta_mu0);
}

template<class FType>
void MacauPrior<FType>::addSideInfo(std::unique_ptr<FType> &Fmat, bool comp_FtF)
{
   // side information
   F = std::move(Fmat);
   use_FtF = comp_FtF;

   // initial value (should be determined automatically)
   lambda_beta = 5.0;
   // Hyper-prior for lambda_beta (mean 1.0, var of 1e+3):
   lambda_beta_mu0 = 1.0;
   lambda_beta_nu0 = 1e-3;
}

template<class FType>
double MacauPrior<FType>::getLinkLambda() 
{
   return lambda_beta; 
}

template<class FType>
const Eigen::VectorXd MacauPrior<FType>::getMu(int n) const 
{
   return this->mu + Uhat.col(n); 
}

template<class FType>
void MacauPrior<FType>::compute_Ft_y_omp(MatrixXd &Ft_y) 
{
   const int num_feat = beta.cols();

   // Ft_y = (U .- mu + Normal(0, Lambda^-1)) * F + sqrt(lambda_beta) * Normal(0, Lambda^-1)
   // Ft_y is [ D x F ] matrix
   MatrixXd tmp = (U() + MvNormal_prec_omp(Lambda, num_cols())).colwise() - mu;
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
void MacauPrior<FType>::sample_beta() 
{
   if (use_FtF) sample_beta_direct();
   else         sample_beta_cg();
}

template<class FType>
void MacauPrior<FType>::setLambdaBeta(double lb)
{
   lambda_beta = lb; 
}

template<class FType>
void MacauPrior<FType>::setTol(double t)
{
   tol = t;
}

template<class FType>
void MacauPrior<FType>::save(std::string prefix, std::string suffix) 
{
   NormalPrior::save(prefix, suffix);
   prefix += "-F" + std::to_string(mode);
   write_dense(prefix + "-link" + suffix, this->beta);
}

template<class FType>
void MacauPrior<FType>::restore(std::string prefix, std::string suffix) 
{
   NormalPrior::restore(prefix, suffix);
   prefix += "-F" + std::to_string(mode);
   read_dense(prefix + "-link" + suffix, this->beta);
}

std::ostream &printSideInfo(std::ostream &os, const SparseDoubleFeat &F) 
{
   os << "SparseDouble [" << F.rows() << ", " << F.cols() << "]\n";
   return os;
}

std::ostream &printSideInfo(std::ostream &os, const Eigen::MatrixXd &F) 
{
   os << "DenseDouble [" << F.rows() << ", " << F.cols() << "]\n";
   return os;
}

std::ostream &printSideInfo(std::ostream &os, const SparseFeat &F) 
{
   os << "SparseBinary [" << F.rows() << ", " << F.cols() << "]\n";
   return os;
}

template<class FType>
std::ostream &MacauPrior<FType>::info(std::ostream &os, std::string indent) 
{
   NormalPrior::info(os, indent);
   os << indent << " SideInfo: "; printSideInfo(os, *F); 
   os << indent << " Method: " << (use_FtF ? "Cholesky Decomposition" : "CG Solver") << "\n"; 
   os << indent << " Tol: " << tol << "\n";
   os << indent << " LambdaBeta: " << lambda_beta << "\n";
   return os;
}

template<class FType>
std::ostream &MacauPrior<FType>::status(std::ostream &os, std::string indent) const 
{
   os << indent << "  " << name << ": Beta = " << beta.norm() << "\n";
   return os;
}

// direct method
template<class FType>
void MacauPrior<FType>::sample_beta_direct() 
{
   MatrixXd Ft_y;
   this->compute_Ft_y_omp(Ft_y);
   MatrixXd K(FtF.rows(), FtF.cols());
   K.triangularView<Eigen::Lower>() = FtF;
   K.diagonal().array() += lambda_beta;
   chol_decomp(K);
   chol_solve_t(K, Ft_y);
   beta = Ft_y;
}

namespace smurff{

// specialization for dense matrices --> always direct method */
template<>
void MacauPrior<Eigen::Matrix<double, -1, -1, 0, -1, -1>>::sample_beta_cg() 
{
   not_implemented("Dense Matrix requires direct method");
}

}

// BlockCG solver
template<class FType>
void MacauPrior<FType>::sample_beta_cg() 
{
   MatrixXd Ft_y;
   this->compute_Ft_y_omp(Ft_y);
   solve_blockcg(beta, *F, lambda_beta, Ft_y, tol, 32, 8);
}

namespace smurff{

   template class MacauPrior<SparseFeat>;
   template class MacauPrior<SparseDoubleFeat>;
   template class MacauPrior<Eigen::MatrixXd>;
}

//macau
/*
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <math.h>
#include <iomanip>

#include "mvnormal.h"
#include "macau.h"
#include "chol.h"
#include "linop.h"

#include "truncnorm.h"
extern "C" {
  #include <sparse.h>
}

using namespace std; 
using namespace Eigen;


 // X = A * B
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

// MacauPrior
template<class FType>
void MacauPrior<FType>::init(const int num_latent, std::unique_ptr<FType> &Fmat, bool comp_FtF) {
  mu.resize(num_latent);
  mu.setZero();

  Lambda.resize(num_latent, num_latent);
  Lambda.setIdentity();
  Lambda *= 10;

  // parameters of Inv-Whishart distribution
  WI.resize(num_latent, num_latent);
  WI.setIdentity();
  mu0.resize(num_latent);
  mu0.setZero();
  b0 = 2;
  df = num_latent;

  // side information
  F = std::move(Fmat);
  use_FtF = comp_FtF;
  if (use_FtF) {
    FtF.resize(F->cols(), F->cols());
    At_mul_A(FtF, *F);
  }

  Uhat.resize(num_latent, F->rows());
  Uhat.setZero();

  beta.resize(num_latent, F->cols());
  beta.setZero();

  // initial value (should be determined automatically)
  lambda_beta = 5.0;
  // Hyper-prior for lambda_beta (mean 1.0, var of 1e+3):
  lambda_beta_mu0 = 1.0;
  lambda_beta_nu0 = 1e-3;
}

template<class FType>
void MacauPrior<FType>::sample_latents(Eigen::MatrixXd &U, const Eigen::SparseMatrix<double> &mat, double mean_value,
                    const Eigen::MatrixXd &samples, double alpha, const int num_latent) {
  const int N = U.cols();
#pragma omp parallel for schedule(dynamic, 2)
  for(int n = 0; n < N; n++) {
    // TODO: try moving mu + Uhat.col(n) inside sample_latent for speed
    sample_latent_blas(U, n, mat, mean_value, samples, alpha, mu + Uhat.col(n), Lambda, num_latent);
  }
}

template<class FType>
void MacauPrior<FType>::sample_latents(ProbitNoise& noiseModel, TensorData & data,
                               std::vector< std::unique_ptr<Eigen::MatrixXd> > & samples, int mode, const int num_latent) {
  // TODO:
}

template<class FType>
void MacauPrior<FType>::sample_latents(double noisePrecision,
                                       TensorData & data,
                                       std::vector< std::unique_ptr<Eigen::MatrixXd> > & samples,
                                       int mode,
                                       const int num_latent) {
  auto& sparseMode = (*data.Y)[mode];
  auto& U = samples[mode];
  const int N = U->cols();
  VectorView<Eigen::MatrixXd> view(samples, mode);

#pragma omp parallel for schedule(dynamic, 2)
  for (int n = 0; n < N; n++) {
    Eigen::VectorXd mu2 = mu + Uhat.col(n);
    sample_latent_tensor(U, n, sparseMode, view, data.mean_value, noisePrecision, mu2, Lambda);
  }
}

template<class FType>
void MacauPrior<FType>::update_prior(const Eigen::MatrixXd &U) {
  // residual (Uhat is later overwritten):
  Uhat.noalias() = U - Uhat;
  MatrixXd BBt = A_mul_At_combo(beta);
  // sampling Gaussian
  tie(mu, Lambda) = CondNormalWishart(Uhat, mu0, b0, WI + lambda_beta * BBt, df + beta.cols());
  sample_beta(U);
  compute_uhat(Uhat, *F, beta);
  lambda_beta = sample_lambda_beta(beta, Lambda, lambda_beta_nu0, lambda_beta_mu0);
}

template<class FType>
double MacauPrior<FType>::getLinkNorm() {
  return beta.norm();
}

// Update beta and Uhat
template<class FType>
void MacauPrior<FType>::sample_beta(const Eigen::MatrixXd &U) {
  const int num_feat = beta.cols();
  // Ft_y = (U .- mu + Normal(0, Lambda^-1)) * F + sqrt(lambda_beta) * Normal(0, Lambda^-1)
  // Ft_y is [ D x F ] matrix
  MatrixXd tmp = (U + MvNormal_prec_omp(Lambda, U.cols())).colwise() - mu;
  MatrixXd Ft_y = A_mul_B(tmp, *F) + sqrt(lambda_beta) * MvNormal_prec_omp(Lambda, num_feat);

  if (use_FtF) {
    MatrixXd K(FtF.rows(), FtF.cols());
    K.triangularView<Eigen::Lower>() = FtF;
    for (int i = 0; i < K.cols(); i++) {
      K(i,i) += lambda_beta;
    }
    chol_decomp(K);
    chol_solve_t(K, Ft_y);
    beta = Ft_y;
  } else {
    // BlockCG
    solve_blockcg(beta, *F, lambda_beta, Ft_y, tol, 32, 8);
  }
}

template<class FType>
void MacauPrior<FType>::sample_latents(ProbitNoise & noise, Eigen::MatrixXd &U, const Eigen::SparseMatrix<double> &mat,
                                       double mean_value, const Eigen::MatrixXd &samples, const int num_latent) {
    const int N = U.cols();
#pragma omp parallel for schedule(dynamic, 2)
  for(int n = 0; n < N; n++) {
    // TODO: try moving mu + Uhat.col(n) inside sample_latent for speed
    sample_latent_blas_probit(U, n, mat, mean_value, samples, mu + Uhat.col(n), Lambda, num_latent);
  }

}

template<class FType>
void MacauPrior<FType>::saveModel(std::string prefix) {
  writeToCSVfile(prefix + "-latentmean.csv", mu);
  writeToCSVfile(prefix + "-link.csv", beta);
}

template class MacauPrior<SparseFeat>;
template class MacauPrior<SparseDoubleFeat>;
template class MacauPrior<Eigen::MatrixXd>;
*/