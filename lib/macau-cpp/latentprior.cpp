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

void ILatentPrior::sample_latents() {
#pragma omp parallel for schedule(dynamic, 2)
    for(int n = 0; n < U.cols(); n++) sample_latent(n); 
}



/**
 *  NormalPrior 
 */

NormalPrior::NormalPrior(Eigen::MatrixXd &U, int num_latent)
    : nU(U)
{
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
}

void NormalPrior::update_prior() {
  tie(mu, Lambda) = CondNormalWishart(nU, mu0, b0, WI, df);
}

void NormalPrior::savePriorInfo(std::string prefix) {
  writeToCSVfile(prefix + "-latentmean.csv", mu);
}


/**
 *  SparseNormalPrior 
 */


void SparseNormalPrior::sample_latent(int n)
{
  const VectorXd &mu_u = getMu(n);
  const MatrixXd &Lambda_u = getLambda(n);
  double mean_rating = model.mean_rating;

  MatrixXd MM = Lambda_u;
  VectorXd rr = VectorXd::Zero(num_latent());
  for (SparseMatrix<double>::InnerIterator it(Y, n); it; ++it) {
    auto col = V.col(it.row());
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


/**
 *  DenseNormalPrior 
 */

void DenseNormalPrior::sample_latent(int n)
{
    /*
    auto x = other.CovF * (other.noiseUU * fac.Y.col(n)) + other.CovL * nrandn(num_latent()).matrix();
    double t = noise.sample(n, 0).first;
    fac.U.col(n).noalias() = x;
    fac.UU += x * x.transpose();
    fac.noiseUU += x * (t * x.transpose());
    */
}

/** MacauPrior */
template<class FType>
MacauPrior<FType>::MacauPrior(SparseMF &m, int p, INoiseModel &n)
    : SparseNormalPrior(m, p, n) {}
    
template<class FType>
void MacauPrior<FType>::addSideInfo(std::unique_ptr<FType> &Fmat, bool comp_FtF)
{
    assert((Fmat->rows() == Y.rows()) && "Number of rows in train must be equal to number of rows in features");

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
void MacauPrior<FType>::update_prior() {
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


SpikeAndSlabPrior::SpikeAndSlabPrior(SparseMF &m, int p, INoiseModel &n)
    : SparseLatentPrior(m, p, n)
{
    const int K = num_latent();
    const int D = U.cols();
    assert(D > 0);
    
    //-- prior params
    alpha = ArrayNd::Ones(K);
    Zcol = VectorNd::Zero(K);
    Zkeep = VectorNd::Constant(K, D);
    W2col = VectorNd::Zero(K);
    r = VectorNd::Constant(K,.5);
}

void SpikeAndSlabPrior::sample_latent(int d)
{
    const int K = num_latent();
    auto &W = U; // aliases
    auto &X = V; // aliases
    
    std::default_random_engine generator;
    std::uniform_real_distribution<double> udist(0,1);
    ArrayNd log_alpha = alpha.log();
    ArrayNd log_r = - r.array().log() + (VectorNd::Ones(K) - r).array().log();

    MatrixNNd XX(MatrixNNd::Zero(K,K));
    VectorNd Wcol = W.col(d);
    VectorNd yX(VectorNd::Zero(K));
    for (SparseMatrixD::InnerIterator it(Y,d); it; ++it) {
        double y = it.value() - model.mean_rating;
        auto Xcol = X.col(it.row());
        yX.noalias() += y * Xcol;
        XX.noalias() += Xcol * Xcol.transpose();
    }

    double t = noise.sample(d, 0).first;

    for(unsigned k=0;k<K;++k) {
        double lambda = t * XX(k,k) + alpha(k);
        double mu = t / lambda * (yX(k) - Wcol.transpose() * XX.col(k) + Wcol(k) * XX(k,k));
        double z1 = log_r(k) -  0.5 * (lambda * mu * mu - log(lambda) + log_alpha(k));
        double z = 1 / (1 + exp(z1));
        double p = udist(generator);
        if (Zkeep(k) > 0 && p < z) {
            Zcol(k)++;
            Wcol(k) = mu + randn() / sqrt(lambda);
        } else {
            Wcol(k) = .0;
        }
    }

    W.col(d) = Wcol;
    W2col += Wcol.array().square().matrix();
}

void SpikeAndSlabPrior::savePriorInfo(std::string prefix) {
}


void SpikeAndSlabPrior::update_prior() {
    const int D = U.cols();
    
    r = ( Zcol.array() + prior_beta ) / ( D + prior_beta * D ) ;

    //-- updata alpha K samples from Gamma
    auto ww = W2col.array() / 2 + prior_beta_0;
    auto tmpz = Zcol.array() / 2 + prior_alpha_0 ;
    alpha = tmpz.binaryExpr(ww, [](double a, double b)->double {
            std::default_random_engine generator;
            std::gamma_distribution<double> distribution(a, 1/b);
            return distribution(generator) + 1e-7;
            });

    Zkeep = Zcol.array();
    Zcol.setZero();
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

