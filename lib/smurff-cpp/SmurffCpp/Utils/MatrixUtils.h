#pragma once

#include <limits>

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include <SmurffCpp/Configs/MatrixConfig.h>
#include <SmurffCpp/Configs/NoiseConfig.h>

#include <SmurffCpp/Utils/Error.h>

namespace smurff { namespace matrix_utils {
   // Conversion of MatrixConfig to/from sparse eigen matrix

   Eigen::SparseMatrix<float> sparse_to_eigen(const smurff::MatrixConfig& matrixConfig);
   std::shared_ptr<smurff::MatrixConfig> eigen_to_sparse(const Eigen::SparseMatrix<float> &, smurff::NoiseConfig n = smurff::NoiseConfig(), bool isScarce = false);

   // Conversion of dense data to/from dense eigen matrix

   Eigen::MatrixXf dense_to_eigen(const smurff::MatrixConfig& matrixConfig);

   std::shared_ptr<smurff::MatrixConfig> eigen_to_dense(const Eigen::MatrixXf &, smurff::NoiseConfig n = smurff::NoiseConfig());

   std::ostream& operator << (std::ostream& os, const MatrixConfig& mc);

   bool equals(const Eigen::MatrixXf& m1, const Eigen::MatrixXf& m2, float precision = std::numeric_limits<float>::epsilon());

   bool equals_vector(const Eigen::VectorXf& v1, const Eigen::VectorXf& v2, float precision = std::numeric_limits<float>::epsilon() * 100);
}}
