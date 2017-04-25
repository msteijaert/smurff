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

#include <csr.h>
#include <dsparse.h>
#include "omp_util.h"

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

// to Eigen::Sparse from IJV
void sparseFromIJV(Eigen::SparseMatrix<double> &X, int* rows, int* cols, double* values, int N);
void sparseFromIJ(Eigen::SparseMatrix<double> &X, int* rows, int* cols, int N);

// to Eigen::Sparse from CSR and BCSR
Eigen::SparseMatrix<double> to_eigen(SparseDoubleMatrix &Y);
Eigen::SparseMatrix<double> to_eigen(SparseBinaryMatrix &Y);

void writeToCSVfile(std::string filename, Eigen::MatrixXd matrix);

std::unique_ptr<SparseFeat> load_bcsr(const char* filename);
std::unique_ptr<SparseDoubleFeat> load_csr(const char* filename);

template<class Matrix>
void write_ddm(const char* filename, const Matrix& matrix){
    std::ofstream out(filename,std::ios::out | std::ios::binary | std::ios::trunc);
    typename Matrix::Index rows=matrix.rows(), cols=matrix.cols();
    out.write((char*) (&rows), sizeof(typename Matrix::Index));
    out.write((char*) (&cols), sizeof(typename Matrix::Index));
    out.write((char*) matrix.data(), rows*cols*sizeof(typename Matrix::Scalar) );
    out.close();
}

template<class Matrix>
Matrix read_ddm(const char* filename) {
    Matrix matrix;
    std::ifstream in(filename,std::ios::in | std::ios::binary);
    typename Matrix::Index rows=0, cols=0;
    in.read((char*) (&rows),sizeof(typename Matrix::Index));
    in.read((char*) (&cols),sizeof(typename Matrix::Index));
    matrix.resize(rows, cols);
    in.read( (char *) matrix.data() , rows*cols*sizeof(typename Matrix::Scalar) );
    in.close();
    return matrix;
}

Eigen::MatrixXd sparse_to_dense(SparseBinaryMatrix &in);
Eigen::MatrixXd sparse_to_dense(SparseDoubleMatrix &in);

typedef Eigen::VectorXd VectorNd;
typedef Eigen::MatrixXd MatrixNNd;
typedef Eigen::ArrayXd ArrayNd;

typedef Eigen::SparseMatrix<double> SparseMatrixD;

