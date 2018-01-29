#pragma once

#include "MatrixDataTempl.hpp"

namespace smurff
{
   class ScarceMatrixData : public MatrixDataTempl<Eigen::SparseMatrix<double> >
   {
   private:
      int num_empty[2] = {0,0};

   public:
      ScarceMatrixData(Eigen::SparseMatrix<double> Y);

   public:
      void init_pre() override;
      void center(double global_mean) override;
      double compute_mode_mean_mn(int mode, int pos) override;

      double train_rmse(const SubModel& model) const override;

      std::ostream& info(std::ostream& os, std::string indent) override;

      void get_pnm(const SubModel& model, int mode, int d, Eigen::VectorXd& rr, Eigen::MatrixXd& MM) override;
      void update_pnm(const SubModel& model, int mode) override;

       std::int64_t nna() const override;
   };
}
