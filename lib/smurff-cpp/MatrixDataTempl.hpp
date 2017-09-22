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

         init_pre_mean_centering();
      }

      PVec   dim() const override { return PVec({ static_cast<int>(Y.cols()), static_cast<int>(Y.rows()) }); }
      int    nnz() const override { return Y.nonZeros(); }
      double sum() const override { return Y.sum(); }

      double offset_to_mean(const PVec& pos) const override
      {
              if (getCenterMode() == CenterModeTypes::CENTER_GLOBAL) return getGlobalMean();
         else if (getCenterMode() == CenterModeTypes::CENTER_VIEW)   return getCwiseMean();
         else if (getCenterMode() == CenterModeTypes::CENTER_ROWS)   return mean(1,pos.at(1));
         else if (getCenterMode() == CenterModeTypes::CENTER_COLS)   return mean(0,pos.at(0));
         else if (getCenterMode() == CenterModeTypes::CENTER_NONE)   return .0;
         assert(false);
         return .0;
      }

      double var_total() const override;
      double sumsq(const SubModel& model) const override;

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