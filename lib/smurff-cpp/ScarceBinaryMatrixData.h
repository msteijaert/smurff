#pragma once

#include "ScarceMatrixData.h"
#include "utils.h"

namespace smurff
{
   class ScarceBinaryMatrixData : public ScarceMatrixData
   {
   public:
      ScarceBinaryMatrixData(Eigen::SparseMatrix<double>& Y);

   public:
      void get_pnm(const SubModel& model, int mode, int n, Eigen::VectorXd& rr, Eigen::MatrixXd& MM) override;
      void update_pnm(const SubModel&, int) override;

      int nna() const override;
  };
}