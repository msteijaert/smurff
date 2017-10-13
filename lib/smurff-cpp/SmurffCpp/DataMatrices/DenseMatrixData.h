#pragma once

#include "FullMatrixData.hpp"

namespace smurff
{
   class DenseMatrixData : public FullMatrixData<Eigen::MatrixXd>
   {
   public:
      DenseMatrixData(Eigen::MatrixXd Y);

   public:
      void center(double global_mean) override;
      double train_rmse(const SubModel& model) const override;
   };
}