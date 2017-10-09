#pragma once

#include <chrono>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <memory>

#include "Config.h"
#include "omp_util.h"

//
// GitHub issue #34:
//    https://github.com/ExaScience/smurff/issues/34
//
// Python golden stardard:
//    https://github.com/ExaScience/smurff/blob/master-smurff-merge/python/io/matrix_io.py
//

namespace smurff { namespace matrix_io
{
   enum class MatrixType
   {
      //sparse types
      none,
      sdm,
      sbm,
      mtx,

      //dense types
      csv,
      ddm
   };

   MatrixConfig read_matrix(const std::string& filename);

   MatrixConfig read_dense_float64_bin(std::istream& in);
   MatrixConfig read_dense_float64_csv(std::istream& in);

   MatrixConfig read_sparse_float64_bin(std::istream& in);
   MatrixConfig read_sparse_float64_mtx(std::istream& in);

   MatrixConfig read_sparse_binary_bin(std::istream& in);

   // ===

   void write_matrix(const std::string& filename, const MatrixConfig& matrixConfig);

   void write_dense_float64_bin(std::ostream& out, const MatrixConfig& matrixConfig);
   void write_dense_float64_csv(std::ostream& out, const MatrixConfig& matrixConfig);

   void write_sparse_float64_bin(std::ostream& out, const MatrixConfig& matrixConfig);
   void write_sparse_float64_mtx(std::ostream& out, const MatrixConfig& matrixConfig);

   void write_sparse_binary_bin(std::ostream& out, const MatrixConfig& matrixConfig);

   namespace eigen{
      void read_matrix(const std::string& filename, Eigen::VectorXd& V);

      void read_matrix(const std::string& filename, Eigen::MatrixXd& X);

      void read_matrix(const std::string& filename, Eigen::SparseMatrix<double>& X);

      void read_dense_float64_bin(std::istream& in, Eigen::MatrixXd& X);
      void read_dense_float64_csv(std::istream& in, Eigen::MatrixXd& X);

      void read_sparse_float64_bin(std::istream& in, Eigen::SparseMatrix<double>& X);
      void read_sparse_float64_mtx(std::istream& in, Eigen::SparseMatrix<double>& X);

      void read_sparse_binary_bin(std::istream& in, Eigen::SparseMatrix<double>& X);

      // ===

      void write_matrix(const std::string& filename, const Eigen::MatrixXd& X);

      void write_matrix(const std::string& filename, const Eigen::SparseMatrix<double>& X);

      void write_dense_float64_bin(std::ostream& out, const Eigen::MatrixXd& X);
      void write_dense_float64_csv(std::ostream& out, const Eigen::MatrixXd& X);

      void write_sparse_float64_bin(std::ostream& out, const Eigen::SparseMatrix<double>& X);
      void write_sparse_float64_mtx(std::ostream& out, const Eigen::SparseMatrix<double>& X);

      void write_sparse_binary_bin(std::ostream& out, const Eigen::SparseMatrix<double>& X);
   }
}}