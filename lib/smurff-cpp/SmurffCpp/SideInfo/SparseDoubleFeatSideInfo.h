#pragma once

#include "ISideInfo.h"

#include "SparseDoubleFeat.h"

#include <memory>
#include <Eigen/Sparse>

namespace smurff {

class SparseDoubleFeatSideInfo : public ISideInfo
{
private:
   std::shared_ptr<SparseDoubleFeat> m_side_info;
   Eigen::SparseMatrix<double, Eigen::RowMajor>* matrix_ptr;
   Eigen::SparseMatrix<double, Eigen::ColMajor>* matrix_col_major_ptr;
   Eigen::SparseMatrix<double, Eigen::RowMajor>* matrix_trans_ptr;
public:

   SparseDoubleFeatSideInfo(std::shared_ptr<SparseDoubleFeat> side_info);
   SparseDoubleFeatSideInfo(int rows, int cols, int nnz, int* rows_ptr, int* cols_ptr, double* vals);
   SparseDoubleFeatSideInfo(int rows, int cols, int nnz, int* rows_ptr, int* cols_ptr);
   ~SparseDoubleFeatSideInfo() override;

public:
   int cols() const override;

   int rows() const override;

public:
   std::ostream& print(std::ostream &os) const override;
   
   bool is_dense() const override;

public:
   //linop

   void compute_uhat(Eigen::MatrixXd& uhat, Eigen::MatrixXd& beta) override;

   void At_mul_A(Eigen::MatrixXd& out) override;

   Eigen::MatrixXd A_mul_B(Eigen::MatrixXd& A) override;

   int solve_blockcg(Eigen::MatrixXd& X, double reg, Eigen::MatrixXd& B, double tol, const int blocksize, const int excess, bool throw_on_cholesky_error = false) override;

   Eigen::VectorXd col_square_sum() override;

   void At_mul_Bt(Eigen::VectorXd& Y, const int col, Eigen::MatrixXd& B) override;

   void add_Acol_mul_bt(Eigen::MatrixXd& Z, const int col, Eigen::VectorXd& b) override;

   inline void AtA_mul_B(Eigen::MatrixXd & out, double reg, Eigen::MatrixXd & B, Eigen::MatrixXd & tmp);

   //only for tests
public:
   std::shared_ptr<SparseDoubleFeat> get_features();
};

}
