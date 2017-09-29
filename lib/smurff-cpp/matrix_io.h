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

enum class DenseMatrixType
{
   none,
   csv,
   ddm
};

enum class SparseMatrixType
{
   none,
   sdm,
   sbm,
   mtx
};

struct sparse_vec_iterator {
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

template<typename Matrix>
Eigen::SparseMatrix<double> sparse_to_eigen(Matrix &Y)
{
     Eigen::SparseMatrix<double> out(Y.nrow, Y.ncol);
     sparse_vec_iterator begin(Y, 0);
     sparse_vec_iterator end(Y, Y.nnz);
     out.setFromTriplets(begin, end);
     return out;
}

template<>
Eigen::SparseMatrix<double> sparse_to_eigen<const smurff::MatrixConfig>(const smurff::MatrixConfig& matrixConfig);

template<>
Eigen::SparseMatrix<double> sparse_to_eigen<smurff::MatrixConfig>(smurff::MatrixConfig& matrixConfig);

template<typename Matrix>
Eigen::MatrixXd dense_to_eigen(Matrix &Y)
{
   std::vector<double> Yvalues = Y.getValues();
   return Eigen::Map<Eigen::MatrixXd>(Yvalues.data(), Y.getNRow(), Y.getNCol());
}

class SparseFeat {
  public:
    BinaryCSR M;
    BinaryCSR Mt;

    SparseFeat() {}

    SparseFeat(int nrow, int ncol, long nnz, int* rows, int* cols) {
      new_bcsr(&M,  nnz, nrow, ncol, rows, cols);
      new_bcsr(&Mt, nnz, ncol, nrow, cols, rows);
    }
    virtual ~SparseFeat() {
      free_bcsr( & M);
      free_bcsr( & Mt);
    }
    int nfeat()    const {return M.ncol;}
    int cols()     const {return M.ncol;}
    int nsamples() const {return M.nrow;}
    int rows()     const {return M.nrow;}
};

class SparseDoubleFeat {
  public:
    CSR M;
    CSR Mt;

    SparseDoubleFeat() {}

    SparseDoubleFeat(int nrow, int ncol, long nnz, int* rows, int* cols, double* vals) {
      new_csr(&M,  nnz, nrow, ncol, rows, cols, vals);
      new_csr(&Mt, nnz, ncol, nrow, cols, rows, vals);
    }
    virtual ~SparseDoubleFeat() {
      free_csr( & M);
      free_csr( & Mt);
    }
    int nfeat()    const {return M.ncol;}
    int cols()     const {return M.ncol;}
    int nsamples() const {return M.nrow;}
    int rows()     const {return M.nrow;}
};

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

// to Eigen::Sparse from

void writeToCSVfile(const std::string& filename, const Eigen::MatrixXd& matrix);
void writeToCSVstream(std::ostream& out, const Eigen::MatrixXd& matrix);

void readFromCSVfile(const std::string& filename, Eigen::MatrixXd& matrix);
void readFromCSVstream(std::istream& in, Eigen::MatrixXd& matrix);

std::unique_ptr<SparseFeat> load_bcsr(const char* filename);
std::unique_ptr<SparseDoubleFeat> load_csr(const char* filename);

void write_ddm(const std::string& filename, const Eigen::MatrixXd& matrix);
void write_ddm(std::ostream& out, const Eigen::MatrixXd& matrix);

void read_ddm(const std::string& filename, Eigen::MatrixXd& matrix);
void read_ddm(std::istream& in, Eigen::MatrixXd& matrix);

Eigen::MatrixXd sparse_to_dense(const SparseBinaryMatrix& in);
Eigen::MatrixXd sparse_to_dense(const SparseDoubleMatrix& in);

bool is_matrix_fname(const std::string& fname);
bool is_sparse_fname(const std::string& fname);
bool is_sparse_binary_fname(const std::string& fname);
bool is_dense_fname(const std::string& fname);
bool is_compact_fname(const std::string& fname);

void read_dense(const std::string& fname, Eigen::MatrixXd& X);
void read_dense(std::istream& in, DenseMatrixType denseMatrixType, Eigen::MatrixXd& X);

void read_dense(const std::string& fname, Eigen::VectorXd &);
void read_dense(std::istream& in, DenseMatrixType denseMatrixType, Eigen::VectorXd& V);

void read_sparse(const std::string& fname, Eigen::SparseMatrix<double> &);

smurff::MatrixConfig read_mtx(const std::string& filename);
smurff::MatrixConfig read_mtx(std::istream& in);

smurff::MatrixConfig read_csv(const std::string& filename);
smurff::MatrixConfig read_csv(std::istream& in);

smurff::MatrixConfig read_ddm(const std::string& filename);
smurff::MatrixConfig read_ddm(std::istream& in);

smurff::MatrixConfig read_dense(const std::string& fname);
smurff::MatrixConfig read_dense(std::istream& in, DenseMatrixType denseMatrixType);

smurff::MatrixConfig read_sparse(const std::string& fname);

smurff::MatrixConfig read_matrix(const std::string& fname);

void write_dense(const std::string& fname, const Eigen::MatrixXd&);
void write_dense(std::ostream& out, DenseMatrixType denseMatrixType, const Eigen::MatrixXd&);

//
// GitHub issue #34:
//    https://github.com/ExaScience/smurff/issues/34
//
// Python golden stardard:
//    https://github.com/ExaScience/smurff/blob/master-smurff-merge/python/io/matrix_io.py
//
smurff::MatrixConfig read_dense_float64(const std::string& filename);
smurff::MatrixConfig read_dense_float64(std::istream& in);

smurff::MatrixConfig read_sparse_float64(const std::string& filename);
smurff::MatrixConfig read_sparse_float64(std::istream& in);

smurff::MatrixConfig read_sparse_binary_matrix(const std::string& filename);
smurff::MatrixConfig read_sparse_binary_matrix(std::istream& in);

void write_dense_float64(const std::string& filename, const smurff::MatrixConfig& Y);
void write_dense_float64(std::ostream& out, const smurff::MatrixConfig& Y);

void write_sparse_float64(const std::string& filename, const smurff::MatrixConfig& Y);
void write_sparse_float64(std::ostream& out, const smurff::MatrixConfig& Y);

void write_sparse_binary_matrix(const std::string& filename, const smurff::MatrixConfig& Y);
void write_sparse_binary_matrix(std::ostream& out, const smurff::MatrixConfig& Y);