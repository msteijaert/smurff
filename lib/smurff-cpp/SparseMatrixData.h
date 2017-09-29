#pragma once

#include "FullMatrixData.hpp"

namespace smurff
{
   class SparseMatrixData : public FullMatrixData<Eigen::SparseMatrix<double> >
   {
   public:
      SparseMatrixData(Eigen::SparseMatrix<double> Y);

   public:
      void center(double global_mean) override;
      double train_rmse(const SubModel& model) const override;
  };
}