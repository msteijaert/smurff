#pragma once

#include "FullMatrixData.hpp"

namespace smurff
{
   class SparseMatrixData : public FullMatrixData<Eigen::SparseMatrix<float> >
   {
   public:
      SparseMatrixData(Eigen::SparseMatrix<float> Y);

      void getMuLambda(const SubModel& model, std::uint32_t mode, int d, Eigen::VectorXf& rr, Eigen::MatrixXf& MM) const override;

   public:
      float train_rmse(const SubModel& model) const override;

   public:
      float var_total() const override;
      
      float sumsq(const SubModel& model) const override;
  };
}
