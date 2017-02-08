#include <Eigen/Dense>
#include <Eigen/Sparse>

typedef Eigen::SparseMatrix<double> SparseMatrixD;

Eigen::MatrixXd ones_Ydense(int N, int D, int K);
Eigen::MatrixXd random_Ydense(int N, int D, int K);
SparseMatrixD ones_Ysparse(int N, int D, int K, double s);
SparseMatrixD random_Ysparse(int N, int D, int K, double s);

SparseMatrixD extract(SparseMatrixD &Y, double s);
SparseMatrixD extract(const Eigen::MatrixXd &Yin, double s);

