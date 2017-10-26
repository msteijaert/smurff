#pragma once

#include <memory>

#include "MatrixData.h"

namespace smurff
{
   template<typename YType>
   class MatrixDataTempl : public MatrixData
   {
   private:
      YType m_Y; // eigen matrix with the data

   public:
      MatrixDataTempl(YType Y) 
         : m_Y(Y)
      {
      }

      void init_pre() override
      {
         assert(nrow() > 0 && ncol() > 0);
      }

      PVec<> dim() const override 
      { 
         return PVec<>({ static_cast<int>(Y().rows()), static_cast<int>(Y().cols()) }); 
      }

      int nnz() const override 
      { 
         return Y().nonZeros(); 
      }

      double sum() const override 
      { 
         return Y().sum(); 
      }

      double var_total() const override;
      double sumsq(const SubModel& model) const override;

   public:
      const YType& Y() const
      {
         return m_Y;
      }

      const YType& Yc(int mode) const
      {
         throw std::runtime_error("Not implemented");
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
