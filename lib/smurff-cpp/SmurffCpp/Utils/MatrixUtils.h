#pragma once

#include <limits>

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include <SmurffCpp/Configs/MatrixConfig.h>

#include <SmurffCpp/LibFastSparseDependency.h>

#include <SmurffCpp/Utils/Error.h>

namespace smurff { namespace matrix_utils {
   // Conversion of MatrixConfig to sparse eigen matrix

   Eigen::SparseMatrix<double> sparse_to_eigen(const smurff::MatrixConfig& matrixConfig);

   // Conversion of dense data to dense eigen matrix - do we need it? (sparse eigen matrix can be converted to dense eigen matrix with = operator)

   Eigen::MatrixXd dense_to_eigen(const smurff::MatrixConfig& matrixConfig);

   Eigen::MatrixXd dense_to_eigen(smurff::MatrixConfig& matrixConfig);

   // Conversion of libfastsparse matrices to dense eigen matrix - do we need it?

   Eigen::MatrixXd sparse_to_dense(const SparseBinaryMatrix& in);

   Eigen::MatrixXd sparse_to_dense(const SparseDoubleMatrix& in);

   std::ostream& operator << (std::ostream& os, const MatrixConfig& mc);

   bool equals(const Eigen::MatrixXd& m1, const Eigen::MatrixXd& m2, double precision = std::numeric_limits<double>::epsilon());
}}
