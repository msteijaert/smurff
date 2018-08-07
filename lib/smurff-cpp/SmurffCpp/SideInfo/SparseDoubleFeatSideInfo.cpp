#include "SparseDoubleFeatSideInfo.h"

#include <SmurffCpp/Utils/linop.h>
#include <SmurffCpp/Utils/MatrixUtils.h>
#include <SmurffCpp/Utils/counters.h>
#include <vector>

using namespace smurff;

SparseDoubleFeatSideInfo::SparseDoubleFeatSideInfo(std::shared_ptr<SparseDoubleFeat> side_info)
   : m_side_info(side_info)
{
    matrix_utils::sparse_eigen_struct str = matrix_utils::csr_to_eigen(m_side_info->M);
    matrix_ptr = str.row_major_sparse;
    matrix_col_major_ptr = str.column_major_sparse;
    matrix_trans_ptr = str.transposed_sparse;
}

SparseDoubleFeatSideInfo::SparseDoubleFeatSideInfo(uint64_t rows, uint64_t cols, uint64_t nnz, const uint32_t* rows_ptr, const uint32_t* cols_ptr, const double* vals) {
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

SparseDoubleFeatSideInfo::SparseDoubleFeatSideInfo(uint64_t rows, uint64_t cols, uint64_t nnz, const uint32_t* rows_ptr, const uint32_t* cols_ptr) {
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

SparseDoubleFeatSideInfo::~SparseDoubleFeatSideInfo() {
    delete matrix_ptr;
    delete matrix_col_major_ptr;
    delete matrix_trans_ptr;
}


int SparseDoubleFeatSideInfo::cols() const
{
   return matrix_ptr->cols();
}

int SparseDoubleFeatSideInfo::rows() const
{
   return matrix_ptr->rows();
}

std::ostream& SparseDoubleFeatSideInfo::print(std::ostream &os) const
{
   double percent = 100.8 * (double)matrix_ptr->nonZeros() / (double)matrix_ptr->rows() / (double) matrix_ptr->cols();
   os << "SparseDouble " << matrix_ptr->nonZeros() << " [" << matrix_ptr->rows() << ", " << matrix_ptr->cols() << "] ("
      << percent << "%)" << std::endl;
   return os;
}

bool SparseDoubleFeatSideInfo::is_dense() const
{
   return false;
}

void SparseDoubleFeatSideInfo::compute_uhat(Eigen::MatrixXd& uhat, Eigen::MatrixXd& beta)
{
    COUNTER("compute_uhat");
    uhat = beta * (*matrix_trans_ptr);
}

void SparseDoubleFeatSideInfo::At_mul_A(Eigen::MatrixXd& out)
{
    COUNTER("At_mul_A");
    out = (*matrix_trans_ptr) * (*matrix_ptr);
}

Eigen::MatrixXd SparseDoubleFeatSideInfo::A_mul_B(Eigen::MatrixXd& A)
{
    COUNTER("A_mul_B");
    return (A * (*matrix_ptr));
}

int SparseDoubleFeatSideInfo::solve_blockcg(Eigen::MatrixXd& X, double reg, Eigen::MatrixXd& B, double tol, const int blocksize, const int excess, bool throw_on_cholesky_error)
{
    COUNTER("solve_blockcg");
    return smurff::linop::solve_blockcg(X, *this, reg, B, tol, blocksize, excess, throw_on_cholesky_error);
}

Eigen::VectorXd SparseDoubleFeatSideInfo::col_square_sum()
{
    COUNTER("col_square_sum");
    const int ncol = matrix_ptr->cols();
    Eigen::VectorXd out(ncol);
    const int* column_ptr = matrix_col_major_ptr->outerIndexPtr();
    const double* vals = matrix_col_major_ptr->valuePtr();

    #pragma omp parallel for schedule(guided)
    for (int col = 0; col < ncol; col++) {
        double tmp = 0;
        int i = column_ptr[col];
        int end = column_ptr[col + 1];
        for (; i < end; i++) {
            tmp += vals[i] * vals[i];
        }
        out(col) = tmp;
    }
    return out;
}

// Y = X[:,col]' * B'
void SparseDoubleFeatSideInfo::At_mul_Bt(Eigen::VectorXd& Y, const int col, Eigen::MatrixXd& B)
{
    COUNTER("At_mul_Bt");
    Eigen::MatrixXd out = matrix_trans_ptr->block(col, 0, col + 1, matrix_trans_ptr->cols()) * B.transpose();
    Y = out.transpose();
}

// computes Z += A[:,col] * b', where a and b are vectors
void SparseDoubleFeatSideInfo::add_Acol_mul_bt(Eigen::MatrixXd& Z, const int col, Eigen::VectorXd& b)
{
    COUNTER("add_Acol_mul_bt");
    Eigen::VectorXd bt = b.transpose();
    const int* cols = matrix_col_major_ptr->innerIndexPtr();
    const double* vals = matrix_col_major_ptr->valuePtr();
    int i = matrix_col_major_ptr->outerIndexPtr()[col];
    const int end = matrix_col_major_ptr->outerIndexPtr()[col + 1];
    const int D = bt.size();
   for (; i < end; i++) 
   {
      int c = cols[i];
      for (int d = 0; d < D; d++) 
      {
         Z(d, c) += vals[i] * b(d);
      }
   }
}

std::shared_ptr<SparseDoubleFeat> SparseDoubleFeatSideInfo::get_features()
{
   return m_side_info;
}