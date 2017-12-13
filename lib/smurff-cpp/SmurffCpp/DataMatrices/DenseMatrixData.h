#pragma once

#include "FullMatrixData.hpp"

namespace smurff
{
   class DenseMatrixData : public FullMatrixData<Eigen::MatrixXd>
   {
   public:
      DenseMatrixData(Eigen::MatrixXd Y);
      void getMuLambda(const SubModel& model, std::uint32_t mode, int d, Eigen::VectorXd& rr, Eigen::MatrixXd& MM) const override;

   public:
      double train_rmse(const SubModel& model) const override;

   public:
      double var_total() const override;
      
      double sumsq(const SubModel& model) const override;
   };
}
