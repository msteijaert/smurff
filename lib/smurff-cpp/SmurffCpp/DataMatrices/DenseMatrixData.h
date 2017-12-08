#pragma once

#include "FullMatrixData.hpp"

namespace smurff
{
   class DenseMatrixData : public FullMatrixData<Eigen::MatrixXd>
   {
   public:
      DenseMatrixData(Eigen::MatrixXd Y);

   public:
      double train_rmse(const SubModel& model) const override;

   public:
      double var_total() const override;
      
      double sumsq(const SubModel& model) const override;
   };
}