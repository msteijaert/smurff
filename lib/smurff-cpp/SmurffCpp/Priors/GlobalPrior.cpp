#include "GlobalPrior.h"

//macau probit - used in MacauPrior, NormalPrior
/*
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
    THROWERROR("Cholesky Decomposition failed!");
  }

  rr.noalias() += Lambda_u * mu_u;
  chol.matrixL().solveInPlace(rr);
  for (int i = 0; i < num_latent; i++) {
    rr[i] += randn0();
  }
  chol.matrixU().solveInPlace(rr);
  s.col(mm).noalias() = rr;
}
*/

//macau - used in python interface
/*
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