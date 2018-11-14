#include "DenseSideInfo.h"

#include <SmurffCpp/Utils/linop.h>

using namespace smurff;

DenseSideInfo::DenseSideInfo(std::shared_ptr<Eigen::MatrixXd> side_info)
   : m_side_info(side_info)
{
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

void DenseSideInfo::compute_uhat(Eigen::MatrixXd& uhat, Eigen::MatrixXd& beta)
{
   uhat = beta * m_side_info->transpose();
}

void DenseSideInfo::At_mul_A(Eigen::MatrixXd& out)
{
   out = m_side_info->transpose() * *m_side_info;
}

Eigen::MatrixXd DenseSideInfo::A_mul_B(Eigen::MatrixXd& A)
{
   return A * *m_side_info;
}

int DenseSideInfo::solve_blockcg(Eigen::MatrixXd& X, double reg, Eigen::MatrixXd& B, double tol, const int blocksize, const int excess, bool throw_on_cholesky_error)
{
   return smurff::linop::solve_blockcg(X, *m_side_info, reg, B, tol, blocksize, excess, throw_on_cholesky_error);
}

Eigen::VectorXd DenseSideInfo::col_square_sum()
{
    return m_side_info->array().square().colwise().sum();
}


void DenseSideInfo::At_mul_Bt(Eigen::VectorXd& Y, const int col, Eigen::MatrixXd& B)
{
   Y = B * m_side_info->col(col);
}

void DenseSideInfo::add_Acol_mul_bt(Eigen::MatrixXd& Z, const int col, Eigen::VectorXd& b)
{
   Z += (m_side_info->col(col) * b.transpose()).transpose();
}

std::shared_ptr<Eigen::MatrixXd> DenseSideInfo::get_features()
{
   return m_side_info;
}
