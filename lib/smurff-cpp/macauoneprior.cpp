#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <math.h>
#include <iostream>

#include "mvnormal.h"
#include "linop.h"
#include "utils.h"
#include "session.h"
#include "macauoneprior.h"
#include "data.h"

using namespace std; 
using namespace Eigen;
namespace smurff {

template<class FType>
MacauOnePrior<FType>::MacauOnePrior(BaseSession &m, int p)
    : ILatentPrior(m, p), Yc(dynamic_cast<ScarceMatrixData &>(m.data()).Yc.at(p))
{
  // parameters of Normal-Gamma distributions
  mu     = VectorXd::Constant(num_latent(), 0.0);
  lambda = VectorXd::Constant(num_latent(), 10.0);
  // hyperparameter (lambda_0)
  l0 = 2.0;
  lambda_a0 = 1.0;
  lambda_b0 = 1.0;
}

template<class FType>
void MacauOnePrior<FType>::addSideInfo(std::unique_ptr<FType> &Fmat, bool) {
  // side information
  F       = std::move(Fmat);
  F_colsq = col_square_sum(*F);

  Uhat = MatrixXd::Constant(num_latent(), F->rows(), 0.0);
  beta = MatrixXd::Constant(num_latent(), F->cols(), 0.0);

  // initial value (should be determined automatically)
  // Hyper-prior for lambda_beta (mean 1.0):
  lambda_beta     = VectorXd::Constant(num_latent(), 5.0);
  lambda_beta_a0 = 0.1;
  lambda_beta_b0 = 0.1;
}

template<class FType>
void MacauOnePrior<FType>::sample_latent(int i)
{
    const int K = num_latent();
    auto &Us = U();
    auto &Vs = V();

    const int nnz = Yc.outerIndexPtr()[i + 1] - Yc.outerIndexPtr()[i];
    VectorXd Yhat(nnz);

    // precalculating Yhat and Qi
    int idx = 0;
    VectorXd Qi = lambda;
    for (SparseMatrix<double>::InnerIterator it(Yc, i); it; ++it, idx++) {
      double alpha = noise().getAlpha();
      Qi.noalias() += alpha * Vs.col(it.row()).cwiseAbs2();
      Yhat(idx)     = Us.col(i).dot( Vs.col(it.row()) );
    }
    VectorXd rnorms(num_latent());
    bmrandn_single(rnorms);

    for (int d = 0; d < K; d++) {
        // computing Lid
        const double uid = Us(d, i);
        double Lid = lambda(d) * (mu(d) + Uhat(d, i));

        idx = 0;
        for ( SparseMatrix<double>::InnerIterator it(Yc, i); it; ++it, idx++) {
            const double vjd = Vs(d, it.row());
            // L_id += alpha * (Y_ij - k_ijd) * v_jd
            double alpha = noise().getAlpha();
            Lid += alpha * (it.value() - (Yhat(idx) - uid*vjd)) * vjd;
            //std::cout << "U(" << d << ", " << i << "): Lid = " << Lid <<std::endl;
        }
        // Now use Lid and Qid to update uid
        double uid_old = Us(d, i);
        double uid_var = 1.0 / Qi(d);

        // sampling new u_id ~ Norm(Lid / Qid, 1/Qid)
        Us(d, i) = Lid * uid_var + sqrt(uid_var) * rnorms(d);

        // updating Yhat
        double uid_delta = Us(d, i) - uid_old;
        idx = 0;
        for (SparseMatrix<double>::InnerIterator it(Yc, i); it; ++it, idx++) {
            Yhat(idx) += uid_delta * Vs(d, it.row());
        }
    }
}

template<class FType>
void MacauOnePrior<FType>::sample_latents() {
    ILatentPrior::sample_latents();

    sample_mu_lambda(U());
    sample_beta(U());
    compute_uhat(Uhat, *F, beta);
    sample_lambda_beta();
}

template<class FType>
void MacauOnePrior<FType>::sample_mu_lambda(const Eigen::MatrixXd &U) {
  MatrixXd Lambda(num_latent(), num_latent());
  MatrixXd WI(num_latent(), num_latent());
  WI.setIdentity();
  int N = U.cols();

  MatrixXd Udelta(num_latent(), N);
#pragma omp parallel for schedule(static)
  for (int i = 0; i < N; i++) {
    for (int d = 0; d < num_latent(); d++) {
      Udelta(d, i) = U(d, i) - Uhat(d, i);
    }
  }
  tie(mu, Lambda) = CondNormalWishart(Udelta, VectorXd::Constant(num_latent(), 0.0), 2.0, WI, num_latent());
  lambda = Lambda.diagonal();
}

template<class FType>
void MacauOnePrior<FType>::sample_beta(const Eigen::MatrixXd &U) {
  // updating beta and beta_var
  const int nfeat = beta.cols();
  const int N = U.cols();
  const int blocksize = 4;

  MatrixXd Z;

#pragma omp parallel for private(Z) schedule(static, 1)
  for (int dstart = 0; dstart < num_latent(); dstart += blocksize) {
    const int dcount = std::min(blocksize, num_latent() - dstart);
    Z.resize(dcount, U.cols());

    for (int i = 0; i < N; i++) {
      for (int d = 0; d < dcount; d++) {
        int dx = d + dstart;
        Z(d, i) = U(dx, i) - mu(dx) - Uhat(dx, i);
      }
    }

    for (int f = 0; f < nfeat; f++) {
      VectorXd zx(dcount), delta_beta(dcount), randvals(dcount);
      // zx = Z[dstart : dstart + dcount, :] * F[:, f]
      At_mul_Bt(zx, *F, f, Z);
      // TODO: check if sampling randvals for whole [nfeat x dcount] matrix works faster
      bmrandn_single( randvals );

      for (int d = 0; d < dcount; d++) {
        int dx = d + dstart;
        double A_df     = lambda_beta(dx) + lambda(dx) * F_colsq(f);
        double B_df     = lambda(dx) * (zx(d) + beta(dx,f) * F_colsq(f));
        double A_inv    = 1.0 / A_df;
        double beta_new = B_df * A_inv + sqrt(A_inv) * randvals(d);
        delta_beta(d)   = beta(dx,f) - beta_new;

        beta(dx, f)     = beta_new;
      }
      // Z[dstart : dstart + dcount, :] += F[:, f] * delta_beta'
      add_Acol_mul_bt(Z, *F, f, delta_beta);
    }
  }
}

template<class FType>
void MacauOnePrior<FType>::sample_lambda_beta() {
  double lambda_beta_a = lambda_beta_a0 + beta.cols() / 2.0;
  VectorXd lambda_beta_b = VectorXd::Constant(beta.rows(), lambda_beta_b0);
  const int D = beta.rows();
  const int F = beta.cols();
#pragma omp parallel
  {
    VectorXd tmp(D);
    tmp.setZero();
#pragma omp for schedule(static)
    for (int f = 0; f < F; f++) {
      for (int d = 0; d < D; d++) {
        tmp(d) += square(beta(d, f));
      }
    }
#pragma omp critical
    {
      lambda_beta_b += tmp / 2;
    }
  }
  for (int d = 0; d < D; d++) {
    lambda_beta(d) = rgamma(lambda_beta_a, 1.0 / lambda_beta_b(d));
  }
}

template<class FType>
void MacauOnePrior<FType>::save(std::string prefix, std::string suffix) {
  write_dense(prefix + "-latentmean" + suffix, mu);
  prefix += "-F" + std::to_string(mode);
  write_dense(prefix + "-link" + suffix, beta);
}

template<class FType>
void MacauOnePrior<FType>::restore(std::string prefix, std::string suffix) {
  read_dense(prefix + "-latentmean" + suffix, mu);
  prefix += "-F" + std::to_string(mode);
  read_dense(prefix + "-link" + suffix, beta);
}

template<class FType>
std::ostream &MacauOnePrior<FType>::status(std::ostream &os, std::string indent) const {
    os << indent << "  " << name << ": Beta = " << beta.norm() << "\n";
    return os;
}

template class MacauOnePrior<SparseFeat>;
template class MacauOnePrior<SparseDoubleFeat>;
template class MacauOnePrior<Eigen::MatrixXd>;

} // end namespace smurff
