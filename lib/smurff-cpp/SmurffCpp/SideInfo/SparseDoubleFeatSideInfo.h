#pragma once

#include "ISideInfo.h"

#include "SparseDoubleFeat.h"

#include <memory>
#include <Eigen/Sparse>

namespace smurff {

class SparseDoubleFeatSideInfo : public ISideInfo
{

public:
   Eigen::SparseMatrix<double, Eigen::RowMajor>* matrix_ptr;
   Eigen::SparseMatrix<double, Eigen::ColMajor>* matrix_col_major_ptr;
   Eigen::SparseMatrix<double, Eigen::RowMajor>* matrix_trans_ptr;

   SparseDoubleFeatSideInfo(uint64_t rows, uint64_t cols, uint64_t nnz, const uint32_t* rows_ptr, const uint32_t* cols_ptr, const double* vals);
   SparseDoubleFeatSideInfo(uint64_t rows, uint64_t cols, uint64_t nnz, const uint32_t* rows_ptr, const uint32_t* cols_ptr);
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

   //only for tests
public:
   std::shared_ptr<SparseDoubleFeat> get_features();
};

}
