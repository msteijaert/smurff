#pragma once

#include "MatrixDataTempl.hpp"

namespace smurff
{
   class ScarceMatrixData : public MatrixDataTempl<Eigen::SparseMatrix<float> >
   {
   private:
      int num_empty[2] = {0,0};

   public:
      ScarceMatrixData(Eigen::SparseMatrix<float> Y);

   public:
      void init_pre() override;
      
      float train_rmse(const SubModel& model) const override;

      std::ostream& info(std::ostream& os, std::string indent) override;

      void getMuLambda(const SubModel& model, std::uint32_t mode, int d, Eigen::VectorXf& rr, Eigen::MatrixXf& MM) const override;
      void update_pnm(const SubModel& model, std::uint32_t mode) override;

      std::uint64_t nna() const override;

   public:
      float var_total() const override;
      
      float sumsq(const SubModel& model) const override;
   };
}
