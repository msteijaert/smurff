#pragma once

#include <memory>

#include <Eigen/Dense>

#include <SmurffCpp/Configs/MatrixConfig.h>

#include "ISideInfo.h"


namespace smurff {

   class DenseSideInfo : public ISideInfo
   {
   private:
      std::shared_ptr<Eigen::MatrixXf> m_side_info;

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

      void compute_uhat(Eigen::MatrixXf& uhat, Eigen::MatrixXf& beta) override;

      void At_mul_A(Eigen::MatrixXf& out) override;

      Eigen::MatrixXf A_mul_B(Eigen::MatrixXf& A) override;

      int solve_blockcg(Eigen::MatrixXf& X, float reg, Eigen::MatrixXf& B, float tol, const int blocksize, const int excess, bool throw_on_cholesky_error = false) override;

      Eigen::VectorXf col_square_sum() override;

      void At_mul_Bt(Eigen::VectorXf& Y, const int col, Eigen::MatrixXf& B) override;

      void add_Acol_mul_bt(Eigen::MatrixXf& Z, const int col, Eigen::VectorXf& b) override;

      //only for tests
   public:
      std::shared_ptr<Eigen::MatrixXf> get_features();
   };

}
