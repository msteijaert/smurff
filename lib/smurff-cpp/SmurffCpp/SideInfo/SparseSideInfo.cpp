#include "SparseSideInfo.h"

#include <SmurffCpp/Utils/linop.h>
#include <SmurffCpp/Utils/MatrixUtils.h>
#include <SmurffCpp/Utils/counters.h>
#include <vector>

using namespace smurff;

SparseSideInfo::SparseSideInfo(uint64_t rows, uint64_t cols, uint64_t nnz, const uint32_t* rows_ptr, const uint32_t* cols_ptr, const double* vals) {
    std::vector<Eigen::Triplet<double>>* triplets = new std::vector<Eigen::Triplet<double>>();
    std::vector<Eigen::Triplet<double>>* triplets_trans = new std::vector<Eigen::Triplet<double>>();
    for (int i = 0; i < nnz; i++) {
        triplets->push_back(Eigen::Triplet<double>(rows_ptr[i], cols_ptr[i], vals[i]));
        triplets_trans->push_back(Eigen::Triplet<double>(cols_ptr[i], rows_ptr[i], vals[i]));
    }

    matrix_ptr = new Eigen::SparseMatrix<double, Eigen::RowMajor>(rows, cols);
    matrix_ptr->setFromTriplets(triplets->begin(), triplets->end());
    matrix_col_major_ptr = new Eigen::SparseMatrix<double, Eigen::ColMajor>(rows, cols);
    matrix_col_major_ptr->setFromTriplets(triplets->begin(), triplets->end());
    matrix_trans_ptr = new Eigen::SparseMatrix<double, Eigen::RowMajor>(cols, rows);
    matrix_trans_ptr->setFromTriplets(triplets_trans->begin(), triplets_trans->end());

    
    delete triplets;
    delete triplets_trans;
}

SparseSideInfo::SparseSideInfo(uint64_t rows, uint64_t cols, uint64_t nnz, const uint32_t* rows_ptr, const uint32_t* cols_ptr) {
    std::vector<Eigen::Triplet<double>>* triplets = new std::vector<Eigen::Triplet<double>>();
    std::vector<Eigen::Triplet<double>>* triplets_trans = new std::vector<Eigen::Triplet<double>>();
    for (int i = 0; i < nnz; i++) {
        triplets->push_back(Eigen::Triplet<double>(rows_ptr[i], cols_ptr[i], 1.0));
        triplets_trans->push_back(Eigen::Triplet<double>(cols_ptr[i], rows_ptr[i], 1.0));
    }

    matrix_ptr = new Eigen::SparseMatrix<double, Eigen::RowMajor>(rows, cols);
    matrix_ptr->setFromTriplets(triplets->begin(), triplets->end());
    matrix_col_major_ptr = new Eigen::SparseMatrix<double, Eigen::ColMajor>(rows, cols);
    matrix_col_major_ptr->setFromTriplets(triplets->begin(), triplets->end());
    matrix_trans_ptr = new Eigen::SparseMatrix<double, Eigen::RowMajor>(cols, rows);
    matrix_trans_ptr->setFromTriplets(triplets_trans->begin(), triplets_trans->end());
    
    delete triplets;
    delete triplets_trans;
}

SparseSideInfo::~SparseSideInfo() {
    delete matrix_ptr;
    delete matrix_col_major_ptr;
    delete matrix_trans_ptr;
}


int SparseSideInfo::cols() const
{
   return matrix_ptr->cols();
}

int SparseSideInfo::rows() const
{
   return matrix_ptr->rows();
}

std::ostream& SparseSideInfo::print(std::ostream &os) const
{
   double percent = 100.8 * (double)matrix_ptr->nonZeros() / (double)matrix_ptr->rows() / (double) matrix_ptr->cols();
   os << "SparseDouble " << matrix_ptr->nonZeros() << " [" << matrix_ptr->rows() << ", " << matrix_ptr->cols() << "] ("
      << percent << "%)" << std::endl;
   return os;
}

bool SparseSideInfo::is_dense() const
{
   return false;
}

void SparseSideInfo::compute_uhat(Eigen::MatrixXd& uhat, Eigen::MatrixXd& beta)
{
    COUNTER("compute_uhat");
    uhat = beta * (*matrix_trans_ptr);
}

void SparseSideInfo::At_mul_A(Eigen::MatrixXd& out)
{
    COUNTER("At_mul_A");
    out = (*matrix_trans_ptr) * (*matrix_ptr);
}

Eigen::MatrixXd SparseSideInfo::A_mul_B(Eigen::MatrixXd& A)
{
    COUNTER("A_mul_B");
    return (A * (*matrix_ptr));
}

int SparseSideInfo::solve_blockcg(Eigen::MatrixXd& X, double reg, Eigen::MatrixXd& B, double tol, const int blocksize, const int excess, bool throw_on_cholesky_error)
{
    COUNTER("solve_blockcg");
    return smurff::linop::solve_blockcg(X, *this, reg, B, tol, blocksize, excess, throw_on_cholesky_error);
}

Eigen::VectorXd SparseSideInfo::col_square_sum()
{
    COUNTER("col_square_sum");
    // component-wise square
    auto E = matrix_col_major_ptr->unaryExpr([](const double &d) { return d * d; });
    // col-wise sum
    return E.transpose() * Eigen::VectorXd::Ones(E.cols());
}

// Y = X[:,col]' * B'
void SparseSideInfo::At_mul_Bt(Eigen::VectorXd& Y, const int col, Eigen::MatrixXd& B)
{
    COUNTER("At_mul_Bt");
    auto out = matrix_trans_ptr->block(col, 0, col + 1, matrix_trans_ptr->cols()) * B.transpose();
    Y = out.transpose();
}

// computes Z += A[:,col] * b', where a and b are vectors
void SparseSideInfo::add_Acol_mul_bt(Eigen::MatrixXd& Z, const int col, Eigen::VectorXd& b)
{
    COUNTER("add_Acol_mul_bt");
    Z += (matrix_col_major_ptr->col(col) * b.transpose()).transpose();
}