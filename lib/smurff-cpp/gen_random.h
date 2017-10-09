#include <random>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "Config.h"

typedef Eigen::SparseMatrix<double> SparseMatrixD;

Eigen::MatrixXd ones_Ydense(int N, int D, int K);
Eigen::MatrixXd random_Ydense(int N, int D, int K);
SparseMatrixD ones_Ysparse(int N, int D, int K, double s);
SparseMatrixD random_Ysparse(int N, int D, int K, double s);

SparseMatrixD extract(SparseMatrixD &Y, double s, std::default_random_engine::result_type seed = std::default_random_engine::default_seed);
SparseMatrixD extract(const Eigen::MatrixXd &Yin, double s, std::default_random_engine::result_type seed = std::default_random_engine::default_seed);

smurff::MatrixConfig extract(smurff::MatrixConfig &, double s, bool rm, std::default_random_engine::result_type seed = std::default_random_engine::default_seed);
smurff::MatrixConfig extract(smurff::MatrixConfig &in, double s, std::default_random_engine::result_type seed = std::default_random_engine::default_seed);