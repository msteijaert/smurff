#pragma once

#include <memory>

#include <Eigen/Dense>

#include <SmurffCpp/Configs/MatrixConfig.h>

#include "ISideInfo.h"


namespace smurff {

   class DenseSideInfo : public ISideInfo
   {
   private:
      std::shared_ptr<Eigen::MatrixXd> m_side_info;

   public:
      DenseSideInfo(const std::shared_ptr<MatrixConfig> &);

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
      std::shared_ptr<Eigen::MatrixXd> get_features();
   };

}
