#include "SparseDoubleFeatSideInfo.h"

#include <SmurffCpp/Utils/linop.h>
#include <SmurffCpp/Utils/MatrixUtils.h>

using namespace smurff;

SparseDoubleFeatSideInfo::SparseDoubleFeatSideInfo(std::shared_ptr<SparseDoubleFeat> side_info)
   : m_side_info(side_info)
{
    matrix_utils::sparse_eigen_struct str = matrix_utils::csr_to_eigen(m_side_info->M);
    matrix_ptr = str.row_major_sparse;
    matrix_trans_ptr = str.column_major_sparse;
}

SparseDoubleFeatSideInfo::~SparseDoubleFeatSideInfo() {
    delete matrix_ptr;
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
   double percent = 100.8 * (double)m_side_info->nnz() / (double)m_side_info->rows() / (double) m_side_info->cols();
   os << "SparseDouble " << m_side_info->nnz() << " [" << m_side_info->rows() << ", " << m_side_info->cols() << "] ("
      << percent << "%)" << std::endl;
   return os;
}

bool SparseDoubleFeatSideInfo::is_dense() const
{
   return false;
}

void SparseDoubleFeatSideInfo::compute_uhat(Eigen::MatrixXd& uhat, Eigen::MatrixXd& beta)
{
   uhat = (*matrix_ptr * beta.transpose()).transpose();
}

void SparseDoubleFeatSideInfo::At_mul_A(Eigen::MatrixXd& out)
{
    out = matrix_ptr->transpose() * (*matrix_ptr);
}

Eigen::MatrixXd SparseDoubleFeatSideInfo::A_mul_B(Eigen::MatrixXd& A)
{
    return (A * (*matrix_ptr));
}

int SparseDoubleFeatSideInfo::solve_blockcg(Eigen::MatrixXd& X, double reg, Eigen::MatrixXd& B, double tol, const int blocksize, const int excess, bool throw_on_cholesky_error)
{
   return smurff::linop::solve_blockcg(X, *m_side_info, reg, B, tol, blocksize, excess, throw_on_cholesky_error);
}

Eigen::VectorXd SparseDoubleFeatSideInfo::col_square_sum()
{
    const int ncol = matrix_ptr->cols();
    Eigen::VectorXd out(ncol);
    const int* column_ptr = matrix_trans_ptr->outerIndexPtr();
    const double* vals = matrix_trans_ptr->valuePtr();

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
   Eigen::MatrixXd out = matrix_ptr->block(0, col, matrix_ptr->rows(), col + 1).transpose() * B.transpose();
   Y = out.transpose();
}

// computes Z += A[:,col] * b', where a and b are vectors
void SparseDoubleFeatSideInfo::add_Acol_mul_bt(Eigen::MatrixXd& Z, const int col, Eigen::VectorXd& b)
{
    Eigen::VectorXd bt = b.transpose();
    const int* cols = matrix_trans_ptr->innerIndexPtr();
    const double* vals = matrix_trans_ptr->valuePtr();
    int i = matrix_trans_ptr->outerIndexPtr()[col];
    const int end = matrix_trans_ptr->outerIndexPtr()[col + 1];
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
