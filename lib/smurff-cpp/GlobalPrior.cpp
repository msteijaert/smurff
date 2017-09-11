#include "GlobalPrior.h"

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
*/

//macau
/*
void sample_latent_tensor(std::unique_ptr<Eigen::MatrixXd> &U,
                          int n,
                          std::unique_ptr<SparseMode> & sparseMode,
                          VectorView<Eigen::MatrixXd> & view,
                          double mean_value,
                          double alpha,
                          Eigen::VectorXd & mu,
                          Eigen::MatrixXd & Lambda) {
  const int nmodes1 = view.size();
  const int num_latent = U->rows();

  MatrixXd MM(num_latent, num_latent);
  MM = Lambda;
  VectorXd rr = VectorXd::Zero(mu.size());

  Eigen::VectorXi & row_ptr = sparseMode->row_ptr;
  Eigen::MatrixXi & indices = sparseMode->indices;
  Eigen::VectorXd & values  = sparseMode->values;

  Eigen::MatrixXd* S0 = view.get(0);

  for (int j = row_ptr(n); j < row_ptr(n + 1); j++) {
    VectorXd col = S0->col(indices(j, 0));
    for (int m = 1; m < nmodes1; m++) {
      col.noalias() = col.cwiseProduct(view.get(m)->col(indices(j, m)));
    }

    MM.triangularView<Eigen::Lower>() += alpha * col * col.transpose();
    rr.noalias() += col * ((values(j) - mean_value) * alpha);
  }

  Eigen::LLT<MatrixXd> chol = MM.llt();
  if(chol.info() != Eigen::Success) {
    throw std::runtime_error("Cholesky Decomposition failed!");
  }

  rr.noalias() += Lambda * mu;
  chol.matrixL().solveInPlace(rr);
  for (int i = 0; i < num_latent; i++) {
    rr[i] += randn0();
  }
  chol.matrixU().solveInPlace(rr);
  U->col(n).noalias() = rr;
}

// global function
void sample_latent(MatrixXd &s, int mm, const SparseMatrix<double> &mat, double mean_rating,
    const MatrixXd &samples, double alpha, const VectorXd &mu_u, const MatrixXd &Lambda_u,
    const int num_latent)
{
  // TODO: add cholesky update version
  MatrixXd MM = MatrixXd::Zero(num_latent, num_latent);
  VectorXd rr = VectorXd::Zero(num_latent);
  for (SparseMatrix<double>::InnerIterator it(mat, mm); it; ++it) {
    auto col = samples.col(it.row());
    MM.noalias() += col * col.transpose();
    rr.noalias() += col * ((it.value() - mean_rating) * alpha);
  }

  Eigen::LLT<MatrixXd> chol = (Lambda_u + alpha * MM).llt();
  if(chol.info() != Eigen::Success) {
    throw std::runtime_error("Cholesky Decomposition failed!");
  }

  rr.noalias() += Lambda_u * mu_u;
  chol.matrixL().solveInPlace(rr);
  for (int i = 0; i < num_latent; i++) {
    rr[i] += randn0();
  }
  chol.matrixU().solveInPlace(rr);
  s.col(mm).noalias() = rr;
}

void sample_latent_blas(MatrixXd &s, int mm, const SparseMatrix<double> &mat, double mean_rating,
    const MatrixXd &samples, double alpha, const VectorXd &mu_u, const MatrixXd &Lambda_u,
    const int num_latent)
{
  MatrixXd MM = Lambda_u;
  VectorXd rr = VectorXd::Zero(num_latent);
  for (SparseMatrix<double>::InnerIterator it(mat, mm); it; ++it) {
    auto col = samples.col(it.row());
    MM.triangularView<Eigen::Lower>() += alpha * col * col.transpose();
    rr.noalias() += col * ((it.value() - mean_rating) * alpha);
  }

  Eigen::LLT<MatrixXd> chol = MM.llt();
  if(chol.info() != Eigen::Success) {
    throw std::runtime_error("Cholesky Decomposition failed!");
  }

  rr.noalias() += Lambda_u * mu_u;
  chol.matrixL().solveInPlace(rr);
  for (int i = 0; i < num_latent; i++) {
    rr[i] += randn0();
  }
  chol.matrixU().solveInPlace(rr);
  s.col(mm).noalias() = rr;
}

void sample_latent_blas_probit(MatrixXd &s, int mm, const SparseMatrix<double> &mat, double mean_rating,
    const MatrixXd &samples, const VectorXd &mu_u, const MatrixXd &Lambda_u,
    const int num_latent)
{ 
    MatrixXd MM = Lambda_u;
    VectorXd rr = VectorXd::Zero(num_latent);
    double z;
    auto u = s.col(mm);
    for (SparseMatrix<double>::InnerIterator it(mat, mm); it; ++it) {
      auto col = samples.col(it.row());
      MM.triangularView<Eigen::Lower>() += col * col.transpose();
			double y = 2 * it.value() - 1;
      z = y * rand_truncnorm(y * col.dot(u), 1.0, 0.0);
      rr.noalias() += col * z;
    }
  Eigen::LLT<MatrixXd> chol = MM.llt();
  if(chol.info() != Eigen::Success) {
    throw std::runtime_error("Cholesky Decomposition failed!");
  }

  rr.noalias() += Lambda_u * mu_u;
  chol.matrixL().solveInPlace(rr);
  for (int i = 0; i < num_latent; i++) {
    rr[i] += randn0();
  }
  chol.matrixU().solveInPlace(rr);
  s.col(mm).noalias() = rr;
}

MacauPrior<Eigen::MatrixXd>* make_dense_prior(int nlatent, double* ptr, int nrows, int ncols, bool colMajor, bool comp_FtF) {
	MatrixXd* Fmat = new MatrixXd(0, 0);
	if (colMajor) {
		*Fmat = Map<Matrix<double, Dynamic, Dynamic, ColMajor> >(ptr, nrows, ncols);
	} else {
		*Fmat = Map<Matrix<double, Dynamic, Dynamic, RowMajor> >(ptr, nrows, ncols);
	}
	unique_ptr<MatrixXd> Fmat_ptr = unique_ptr<MatrixXd>(Fmat);
	return new MacauPrior<MatrixXd>(nlatent, Fmat_ptr, comp_FtF);
}
*/