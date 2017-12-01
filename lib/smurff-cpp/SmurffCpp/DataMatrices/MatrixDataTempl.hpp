#pragma once

#include <memory>

#include "MatrixData.h"

namespace smurff
{
   template<typename YType>
   class MatrixDataTempl : public MatrixData
   {
   private:
      // eigen matrices with the data
      std::shared_ptr<std::vector<YType> > m_Yv;

   public:
      MatrixDataTempl(YType Y)
      {
         m_Yv = std::shared_ptr<std::vector<YType> >(new std::vector<YType>());
         m_Yv->push_back(Y.transpose());
         m_Yv->push_back(Y);
      }

      void init_pre() override
      {
         assert(nrow() > 0 && ncol() > 0);
      }

      PVec<> dim() const override 
      { 
         return PVec<>({ static_cast<int>(Y().rows()), static_cast<int>(Y().cols()) }); 
      }

      std::uint64_t nnz() const override 
      { 
         return Y().nonZeros(); 
      }

      double sum() const override 
      { 
         return Y().sum(); 
      }

   public:
      const YType& Y(int mode = 1) const
      {
         return m_Yv->operator[](mode);
      }
   };
}
