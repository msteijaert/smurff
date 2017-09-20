#pragma once

#include "MatrixData.h"

namespace smurff
{
   template<typename YType>
   class MatrixDataTempl : public MatrixData
   {
   public:
      MatrixDataTempl(YType Y) : Y(Y)
      {
      }

      //init and center
      void init_pre() override
      {
         assert(nrow() > 0 && ncol() > 0);

         Yc.push_back(Y);
         Yc.push_back(Y.transpose());

         cwise_mean = sum() / (size() - nna());
      }

      PVec   dim() const override { return PVec(Y.cols(), Y.rows()); }
      int    nnz() const override { return Y.nonZeros(); }
      double sum() const override { return Y.sum(); }

      double offset_to_mean(const PVec &pos) const override
      {
              if (center_mode == CENTER_GLOBAL) return global_mean;
         else if (center_mode == CENTER_VIEW)   return cwise_mean;
         else if (center_mode == CENTER_ROWS)   return mean(1,pos.at(1));
         else if (center_mode == CENTER_COLS)   return mean(0,pos.at(0));
         else if (center_mode == CENTER_NONE)   return .0;
         assert(false);
         return .0;
      }

      double var_total() const override;
      double sumsq(const SubModel &) const override;

      YType Y;
      std::vector<YType> Yc; // centered versions
   };

   template<>
   double MatrixDataTempl<Eigen::MatrixXd>::var_total() const;

   template<>
   double MatrixDataTempl<Eigen::SparseMatrix<double> >::var_total() const;

   template<>
   double MatrixDataTempl<Eigen::MatrixXd>::sumsq(const SubModel &model) const;

   template<>
   double MatrixDataTempl<Eigen::SparseMatrix<double> >::sumsq(const SubModel &model) const;
}