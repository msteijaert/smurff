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
   smurff::linop::compute_uhat(uhat, *m_side_info, beta);
}

void DenseSideInfo::At_mul_A(Eigen::MatrixXd& out)
{
   smurff::linop::At_mul_A(out, *m_side_info);
}

Eigen::MatrixXd DenseSideInfo::A_mul_B(Eigen::MatrixXd& A)
{
   return smurff::linop::A_mul_B(A, *m_side_info);
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
   smurff::linop::At_mul_Bt(Y, *m_side_info, col, B);
}

void DenseSideInfo::add_Acol_mul_bt(Eigen::MatrixXd& Z, const int col, Eigen::VectorXd& b)
{
   smurff::linop::add_Acol_mul_bt(Z, *m_side_info, col, b);
}

std::shared_ptr<Eigen::MatrixXd> DenseSideInfo::get_features()
{
   return m_side_info;
}
