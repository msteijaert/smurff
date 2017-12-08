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
      
      double train_rmse(const SubModel& model) const override;

      std::ostream& info(std::ostream& os, std::string indent) override;

      void get_pnm(const SubModel& model, std::uint32_t mode, int d, Eigen::VectorXd& rr, Eigen::MatrixXd& MM) override;
      void update_pnm(const SubModel& model, std::uint32_t mode) override;

      std::uint64_t nna() const override;

   public:
      double var_total() const override;
      
      double sumsq(const SubModel& model) const override;
   };
}