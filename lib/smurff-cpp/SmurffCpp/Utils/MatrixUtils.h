#pragma once

#include <limits>

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include <SmurffCpp/Configs/MatrixConfig.h>
#include <SmurffCpp/Configs/NoiseConfig.h>

#include <SmurffCpp/Utils/Error.h>

namespace smurff { namespace matrix_utils {
   // Conversion of MatrixConfig to/from sparse eigen matrix

   Eigen::SparseMatrix<double> sparse_to_eigen(const smurff::MatrixConfig& matrixConfig);
   std::shared_ptr<smurff::MatrixConfig> eigen_to_sparse(const Eigen::SparseMatrix<double> &, smurff::NoiseConfig n = smurff::NoiseConfig(), bool isScarce = false);

   // Conversion of dense data to/from dense eigen matrix

   Eigen::MatrixXd dense_to_eigen(const smurff::MatrixConfig& matrixConfig);

   std::shared_ptr<smurff::MatrixConfig> eigen_to_dense(const Eigen::MatrixXd &, smurff::NoiseConfig n = smurff::NoiseConfig());

   std::ostream& operator << (std::ostream& os, const MatrixConfig& mc);

   bool equals(const Eigen::MatrixXd& m1, const Eigen::MatrixXd& m2, double precision = std::numeric_limits<double>::epsilon());

   bool equals_vector(const Eigen::VectorXd& v1, const Eigen::VectorXd& v2, double precision = std::numeric_limits<double>::epsilon() * 100);
}}
