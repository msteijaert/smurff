#pragma once

#include <iostream>

#include <Eigen/Dense>

namespace smurff {

   class ISideInfo
   {
   public:
      virtual ~ISideInfo() {}

      virtual int cols() const = 0;

      virtual int rows() const = 0;

      virtual std::ostream& print(std::ostream &os) const = 0;

      virtual bool is_dense() const = 0;

   public:
      //linop

      virtual void compute_uhat(Eigen::MatrixXf& uhat, Eigen::MatrixXf& beta) = 0;

      virtual void At_mul_A(Eigen::MatrixXf& out) = 0;

      virtual Eigen::MatrixXf A_mul_B(Eigen::MatrixXf& A) = 0;

      virtual int solve_blockcg(Eigen::MatrixXf& X, float reg, Eigen::MatrixXf& B, float tol, const int blocksize, const int excess, bool throw_on_cholesky_error = false) = 0;

      virtual Eigen::VectorXf col_square_sum() = 0;

      virtual void At_mul_Bt(Eigen::VectorXf& Y, const int col, Eigen::MatrixXf& B) = 0;

      virtual void add_Acol_mul_bt(Eigen::MatrixXf& Z, const int col, Eigen::VectorXf& b) = 0;
   };

}
