#pragma once

#include <limits>

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include <SmurffCpp/Configs/MatrixConfig.h>

#include <SmurffCpp/SideInfo/LibFastSparseDependency.h>

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

   typedef struct {
       Eigen::SparseMatrix<double, Eigen::RowMajor>* row_major_sparse;
       Eigen::SparseMatrix<double, Eigen::ColMajor>* column_major_sparse;
   } sparse_eigen_struct;

   sparse_eigen_struct csr_to_eigen(const CSR& csr);

   std::ostream& operator << (std::ostream& os, const MatrixConfig& mc);

   bool equals(const Eigen::MatrixXd& m1, const Eigen::MatrixXd& m2, double precision = std::numeric_limits<double>::epsilon());

   bool equals_vector(const Eigen::VectorXd& v1, const Eigen::VectorXd& v2, double precision = std::numeric_limits<double>::epsilon() * 100);
}}
