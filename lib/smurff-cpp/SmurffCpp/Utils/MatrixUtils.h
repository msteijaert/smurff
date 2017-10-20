#pragma once

#include <array>
#include <limits>
#include <unordered_map>

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include <SmurffCpp/Configs/MatrixConfig.h>

#include <SmurffCpp/LibFastSparseDependency.h>

namespace smurff { namespace matrix_utils {

   struct sparse_vec_iterator
   {
      sparse_vec_iterator(int *rows, int *cols, int pos)
         : rows(rows), cols(cols), vals(0), fixed_val(1.0), pos(pos) {}
      sparse_vec_iterator(int *rows, int *cols, double *vals, uint64_t pos)
         : rows(rows), cols(cols), vals(vals), fixed_val(NAN), pos(pos) {}
      sparse_vec_iterator(int *rows, int *cols, double fixed_val, uint64_t pos)
         : rows(rows), cols(cols), vals(0), fixed_val(fixed_val), pos(pos) {}
      sparse_vec_iterator(const SparseDoubleMatrix &Y, int pos)
         : rows(Y.rows), cols(Y.cols), vals(Y.vals), fixed_val(NAN), pos(pos) {}
      sparse_vec_iterator(const SparseBinaryMatrix &Y, int pos)
         : rows(Y.rows), cols(Y.cols), vals(0), fixed_val(1.0), pos(pos) {}

   //
   // Commented out this constructor. Cause currently we have a specialized sparse_to_eigen function
   // Might remove it later. But for now it does the job
   //
   // sparse_vec_iterator(const smurff::MatrixConfig &Y, int pos)
   //      : rows(Y.rows), cols(Y.cols), vals(Y.values), fixed_val(Y.binary), pos(pos) {}
   //

      int *rows, *cols;
      double *vals; // can be null pointer -> use fixed value
      double fixed_val;
      int pos;

      bool operator!=(const sparse_vec_iterator &other) const {
         assert(rows == other.rows);
         assert(cols == other.cols);
         assert(vals == other.vals);
         return pos != other.pos;
      }

      sparse_vec_iterator &operator++() { pos++; return *this; }

      typedef Eigen::Triplet<double> T;
      T v;

      T* operator->() {
         // also convert from 1-base to 0-base
         uint32_t row = rows[pos];
         uint32_t col = cols[pos];
         double val = vals ? vals[pos] : 1.0;
         v = T(row, col, val);
         return &v;
      }
   };

   // Conversion of sparse data to sparse eigen matrix - do we need it?

   template<typename Matrix>
   Eigen::SparseMatrix<double> sparse_to_eigen(Matrix &Y)
   {
      Eigen::SparseMatrix<double> out(Y.nrow, Y.ncol);
      sparse_vec_iterator begin(Y, 0);
      sparse_vec_iterator end(Y, Y.nnz);
      out.setFromTriplets(begin, end);
      return out;
   }

   // Conversion of MatrixConfig to sparse eigen matrix

   template<>
   Eigen::SparseMatrix<double> sparse_to_eigen<const smurff::MatrixConfig>(const smurff::MatrixConfig& matrixConfig);

   template<>
   Eigen::SparseMatrix<double> sparse_to_eigen<smurff::MatrixConfig>(smurff::MatrixConfig& matrixConfig);

   // Conversion of TensorConfig to sparse eigen matrix

   template<>
   Eigen::SparseMatrix<double> sparse_to_eigen<const smurff::TensorConfig>(const smurff::TensorConfig& tensorConfig);

   template<>
   Eigen::SparseMatrix<double> sparse_to_eigen<smurff::TensorConfig>(smurff::TensorConfig& tensorConfig);

   // Conversion of dense data to dense eigen matrix - do we need it? (sparse eigen matrix can be converted to dense eigen matrix with = operator)

   template<typename Matrix>
   Eigen::MatrixXd dense_to_eigen(Matrix &Y)
   {
   std::vector<double> Yvalues = Y.getValues();
   return Eigen::Map<Eigen::MatrixXd>(Yvalues.data(), Y.getNRow(), Y.getNCol());
   }

   // Conversion of libfastsparse matrices to dense eigen matrix - do we need it?

   Eigen::MatrixXd sparse_to_dense(const SparseBinaryMatrix& in);

   Eigen::MatrixXd sparse_to_dense(const SparseDoubleMatrix& in);

   // Conversion of tensor config to matrix config

   smurff::MatrixConfig tensor_to_matrix(const smurff::TensorConfig& tensorConfig);

   template <typename Matrix>
   inline bool is_binary(const Matrix &M)
   {
      auto *values = M.valuePtr();
      for(int i=0; i<M.nonZeros(); ++i) {
         if (values[i] != 1.0 && values[i] != 0.0) return false;
      }

      std::cout << "Detected binary matrix\n";

      return true;
   }

   std::ostream& operator << (std::ostream& os, const MatrixConfig& mc);
   std::ostream& operator << (std::ostream& os, const TensorConfig& tc);

   bool equals(const Eigen::MatrixXd& m1, const Eigen::MatrixXd& m2, double precision = std::numeric_limits<double>::epsilon());
   Eigen::MatrixXd slice( const TensorConfig& tensorConfig
                        , const std::array<std::uint64_t, 2> fixedDims
                        , const std::unordered_map<std::uint64_t, std::uint32_t>& dimCoords
                        );

}}