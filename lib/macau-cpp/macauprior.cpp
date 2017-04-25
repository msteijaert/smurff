#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <math.h>
#include <iomanip>

#include "mvnormal.h"
#include "session.h"
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

namespace Macau {

template<class FType>
MacauPrior<FType>::MacauPrior(BaseSession &m, int p)
    : NormalPrior(m, p, "MacauPrior")  {}

template<class FType>
void MacauPrior<FType>::init()
{
    NormalPrior::init();

    assert((F->rows() == num_cols()) && 
            "Number of rows in train must be equal to number of rows in features");

    if (use_FtF) {
        FtF.resize(F->cols(), F->cols());
        At_mul_A(FtF, *F);
    }

    Uhat.resize(this->num_latent(), F->rows());
    Uhat.setZero();

    beta.resize(this->num_latent(), F->cols());
    beta.setZero();
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
void MacauPrior<FType>::sample_latents() {
  NormalPrior::sample_latents();

  assert(num_sys() == 1);
  // residual (Uhat is later overwritten):
  Uhat.noalias() = U(0) - Uhat;
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
    assert(num_sys() == 1);

    // Ft_y = (U .- mu + Normal(0, Lambda^-1)) * F + sqrt(lambda_beta) * Normal(0, Lambda^-1)
    // Ft_y is [ D x F ] matrix
    MatrixXd tmp = (U(0) + MvNormal_prec_omp(Lambda, num_cols())).colwise() - mu;
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
    if (use_FtF) sample_beta_direct();
    else         sample_beta_cg();
}

// specialization for dense matrices --> always direct method */
template<>
void MacauPrior<Eigen::Matrix<double, -1, -1, 0, -1, -1>>::sample_beta_cg() {
    not_implemented("Dense Matrix requires direct method");
}

// direct method
template<class FType>
void MacauPrior<FType>::sample_beta_direct() {
    MatrixXd Ft_y;
    this->compute_Ft_y_omp(Ft_y);
    MatrixXd K(FtF.rows(), FtF.cols());
    K.triangularView<Eigen::Lower>() = FtF;
    K.diagonal().array() += lambda_beta;
    chol_decomp(K);
    chol_solve_t(K, Ft_y);
    beta = Ft_y;

}

// BlockCG solver
template<class FType>
void MacauPrior<FType>::sample_beta_cg() {
    MatrixXd Ft_y;
    this->compute_Ft_y_omp(Ft_y);
    solve_blockcg(beta, *F, lambda_beta, Ft_y, tol, 32, 8);
}

template<class FType>
void MacauPrior<FType>::savePriorInfo(std::string prefix) {
  prefix += "-F" + std::to_string(pos);
  writeToCSVfile(prefix + "-latentmean.csv", this->mu);
  writeToCSVfile(prefix + "-link.csv", this->beta);
}

std::ostream &printSideInfo(std::ostream &os, const SparseDoubleFeat &F) {
    os << "SparseDouble [" << F.rows() << ", " << F.cols() << "]\n";
    return os;
}

std::ostream &printSideInfo(std::ostream &os, const Eigen::MatrixXd &F) {
    os << "DenseDouble [" << F.rows() << ", " << F.cols() << "]\n";
    return os;
}

std::ostream &printSideInfo(std::ostream &os, const SparseFeat &F) {
    os << "SparseBinary [" << F.rows() << ", " << F.cols() << "]\n";
    return os;
}

template<class FType>
std::ostream &MacauPrior<FType>::printInitStatus(std::ostream &os, std::string indent) {
    NormalPrior::printInitStatus(os, indent);
    os << indent << " SideInfo: "; printSideInfo(os, *F); 
    os << indent << " Method: " << (use_FtF ? "Cholesky Decompistion" : "CG Solver") << "\n"; 
    os << indent << " Tol: " << tol << "\n";
    os << indent << " LambdaBeta: " << lambda_beta << "\n";
    return os;
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

template class MacauPrior<SparseFeat>;
template class MacauPrior<SparseDoubleFeat>;
template class MacauPrior<Eigen::MatrixXd>;

} // end namespace Macau

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


