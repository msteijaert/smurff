#pragma once

/* macau
void sample_latent(Eigen::MatrixXd &s,
                   int mm,
                   const Eigen::SparseMatrix<double> &mat,
                   double mean_rating,
                   const Eigen::MatrixXd &samples,
                   double alpha,
                   const Eigen::VectorXd &mu_u,
                   const Eigen::MatrixXd &Lambda_u,
                   const int num_latent);

void sample_latent_blas_probit(Eigen::MatrixXd &s,
                        int mm,
                        const Eigen::SparseMatrix<double> &mat,
                        double mean_rating,
                        const Eigen::MatrixXd &samples,
                        const Eigen::VectorXd &mu_u,
                        const Eigen::MatrixXd &Lambda_u,
                        const int num_latent);
                        
void sample_latent_tensor(std::unique_ptr<Eigen::MatrixXd> &U,
                          int n,
                          std::unique_ptr<SparseMode> & sparseMode,
                          VectorView<Eigen::MatrixXd> & view,
                          double mean_value,
                          double alpha,
                          Eigen::VectorXd & mu,
                          Eigen::MatrixXd & Lambda);

MacauPrior<Eigen::MatrixXd>* make_dense_prior(int nlatent, double* ptr, int nrows, int ncols, bool colMajor, bool comp_FtF);
*/