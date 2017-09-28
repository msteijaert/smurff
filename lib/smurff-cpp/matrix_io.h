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

#include <dsparse.h>
#include <csr.h>

#include "config.h"
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
      none,
      sdm,
      sbm,
      mtx,
      csv,
      ddm
   };

   MatrixConfig read_matrix(const std::string& filename);

   MatrixConfig read_dense_float64_bin(std::istream& in);
   MatrixConfig read_dense_float64_csv(std::istream& in);

   MatrixConfig read_sparse_float64_bin(std::istream& in);
   MatrixConfig read_sparse_float64_mtx(std::istream& in);

   MatrixConfig read_sparse_binary_bin(std::istream& in);

   void write_matrix(const std::string& filename, const MatrixConfig& matrixConfig, MatrixType matrixType);

   void write_dense_float64_bin(std::ostream& out, const MatrixConfig& matrixConfig);
   void write_dense_float64_csv(std::ostream& out, const MatrixConfig& matrixConfig);

   void write_sparse_float64_bin(std::ostream& out, const MatrixConfig& matrixConfig);
   void write_sparse_float64_mtx(std::ostream& out, const MatrixConfig& matrixConfig);

   void write_sparse_binary_bin(std::ostream& out, const MatrixConfig& matrixConfig);
}}