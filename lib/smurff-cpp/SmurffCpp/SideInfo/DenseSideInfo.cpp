#include "DenseSideInfo.h"

#include <SmurffCpp/Utils/linop.h>

using namespace smurff;

DenseSideInfo::DenseSideInfo(const std::shared_ptr<MatrixConfig> &side_info)
{
   m_side_info = std::make_shared<Eigen::MatrixXf>(matrix_utils::dense_to_eigen(*side_info));
}

int DenseSideInfo::cols() const
{
   return m_side_info->cols();
}

int DenseSideInfo::rows() const
{
   return m_side_info->rows();
}

std::ostream& DenseSideInfo::print(std::ostream &os) const
{
   os << "DenseDouble [" << m_side_info->rows() << ", " << m_side_info->cols() << "]" << std::endl;
   return os;
}

bool DenseSideInfo::is_dense() const
{
   return true;
}

void DenseSideInfo::compute_uhat(Eigen::MatrixXf& uhat, Eigen::MatrixXf& beta)
{
   uhat = beta * m_side_info->transpose();
}

void DenseSideInfo::At_mul_A(Eigen::MatrixXf& out)
{
   out = m_side_info->transpose() * *m_side_info;
}

Eigen::MatrixXf DenseSideInfo::A_mul_B(Eigen::MatrixXf& A)
{
   return A * *m_side_info;
}

int DenseSideInfo::solve_blockcg(Eigen::MatrixXf& X, float reg, Eigen::MatrixXf& B, float tol, const int blocksize, const int excess, bool throw_on_cholesky_error)
{
   return smurff::linop::solve_blockcg(X, *m_side_info, reg, B, tol, blocksize, excess, throw_on_cholesky_error);
}

Eigen::VectorXf DenseSideInfo::col_square_sum()
{
    return m_side_info->array().square().colwise().sum();
}


void DenseSideInfo::At_mul_Bt(Eigen::VectorXf& Y, const int col, Eigen::MatrixXf& B)
{
   Y = B * m_side_info->col(col);
}

void DenseSideInfo::add_Acol_mul_bt(Eigen::MatrixXf& Z, const int col, Eigen::VectorXf& b)
{
   Z += (m_side_info->col(col) * b.transpose()).transpose();
}

std::shared_ptr<Eigen::MatrixXf> DenseSideInfo::get_features()
{
   return m_side_info;
}
