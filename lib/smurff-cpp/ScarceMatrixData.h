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
      void center(double) override;
      double compute_mode_mean(int,int) override;

      double train_rmse(const SubModel &) const override;

      std::ostream& info(std::ostream& os, std::string indent) override;

      void get_pnm(const SubModel &,int,int,VectorNd &, MatrixNNd &) override;
      void update_pnm(const SubModel &,int) override;

      int nna() const override;
   };
}