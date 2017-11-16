#include "GlobalPrior.h"

//macau tensor - used in MacauPrior, NormalPrior
/*
// U - matrix from Model (for current mode)
// n - index of column in U
// sparseMode - sparse view of Y with reduced mode dimension
// view - vector view all V matrices (matrices from Model other than U)
// mean_value - mean value of Y
// alpha - noise precision
// mu - parameter of distribution
// Lambda - parameter of distribution
void sample_latent_tensor(std::unique_ptr<Eigen::MatrixXd> &U,
                          int n,
                          std::unique_ptr<SparseMode> & sparseMode,
                          VectorView<Eigen::MatrixXd> & view,
                          double mean_value,
                          double alpha,
                          Eigen::VectorXd & mu,
                          Eigen::MatrixXd & Lambda) 
{
  const int nmodes1 = view.size();
  const int num_latent = U->rows();

  // init MM and rr

  MatrixXd MM = MatrixXd::Zero(num_latent(), num_latent());
  VectorXd rr = VectorXd::Zero(mu.size());

  // get_pnm

  Eigen::VectorXi & row_ptr = sparseMode->row_ptr;
  Eigen::MatrixXi & indices = sparseMode->indices;
  Eigen::VectorXd & values  = sparseMode->values;

  Eigen::MatrixXd* S0 = view.get(0);

  for (int j = row_ptr(n); j < row_ptr(n + 1); j++) //j will be the index of row in indices matrix corresponding to plane n in sparse view
  {
    VectorXd col = S0->col(indices(j, 0)); //create a copy of column from m'th V (m = 0)
    for (int m = 1; m < nmodes1; m++) //go through other V matrices
    {
      col.noalias() = col.cwiseProduct(view.get(m)->col(indices(j, m))); //multiply by column from m'th V
    }

    MM.triangularView<Eigen::Lower>() += alpha * col * col.transpose(); // MM = MM + (col * colT) * alpha (where col = product of columns in each V)
    rr.noalias() += col * ((values(j) - mean_value) * alpha); // rr = rr + (col * value) * alpha (where value = j'th value of Y)
  }

  // add hyperparams
  rr.noalias() += Lambda * mu;
  MM.noalias() += Lambda;

  //Solve system of linear equations for x: MM * x = rr

  Eigen::LLT<MatrixXd> chol = MM.llt();
  if(chol.info() != Eigen::Success) 
    throw std::runtime_error("Cholesky Decomposition failed!");

  chol.matrixL().solveInPlace(rr); // solve for y: y = L^-1 * b
  rr.noalias() += nrandn(num_latent());
  chol.matrixU().solveInPlace(rr); // solve for x: x = U^-1 * y

  U->col(n).noalias() = rr; // rr is equal to x
}
*/

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