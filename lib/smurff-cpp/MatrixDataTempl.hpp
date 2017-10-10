#pragma once

#include <memory>

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

         Ycentered = std::shared_ptr<std::vector<YType> >(new std::vector<YType>());
         Ycentered->push_back(Y.transpose());
         Ycentered->push_back(Y);

         init_cwise_mean();
      }

      PVec<>   dim() const override { return PVec<>({ static_cast<int>(Y.rows()), static_cast<int>(Y.cols()) }); }
      int    nnz() const override { return Y.nonZeros(); }
      double sum() const override { return Y.sum(); }

      double offset_to_mean(const PVec<>& pos) const override
      {
              if (getCenterMode() == CenterModeTypes::CENTER_GLOBAL) return getGlobalMean();
         else if (getCenterMode() == CenterModeTypes::CENTER_VIEW)   return getCwiseMean();
         else if (getCenterMode() == CenterModeTypes::CENTER_ROWS)   return getModeMeanItem(1,pos.at(1));
         else if (getCenterMode() == CenterModeTypes::CENTER_COLS)   return getModeMeanItem(0,pos.at(0));
         else if (getCenterMode() == CenterModeTypes::CENTER_NONE)   return .0;
         assert(false);
         return .0;
      }

      double var_total() const override;
      double sumsq(const SubModel& model) const override;

      YType Y; // eigen matrix with the data
      
   private:
      std::shared_ptr<std::vector<YType> > Ycentered; // centered versions of original matrix (transposed, original)

   public:
      const std::vector<YType>& getYc() const
      {
         assert(Ycentered);
         return *Ycentered.get();
      }

      std::shared_ptr<std::vector<YType> > getYcPtr() const
      {
         assert(Ycentered);
         return Ycentered;
      }
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
