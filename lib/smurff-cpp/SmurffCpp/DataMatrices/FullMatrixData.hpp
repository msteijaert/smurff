#pragma once

#include "MatrixDataTempl.hpp"

#include <SmurffCpp/VMatrixExprIterator.hpp>
#include <SmurffCpp/ConstVMatrixExprIterator.hpp>

#include <SmurffCpp/Utils/ThreadVector.hpp>

namespace smurff
{
   template<class YType>
   class FullMatrixData : public MatrixDataTempl<YType>
   {
   protected:
      Eigen::MatrixXf VV[2]; // sum of v * vT, where v is column of V

   public:
      FullMatrixData(YType Y) 
         : MatrixDataTempl<YType>(Y)
      {
         this->name = "MatrixData [fully known]";
      }

   public:
      //purpose of update_pnm is to cache VV matrix
      void update_pnm(const SubModel& model, uint32_t mode) override
      {
         auto Vf = *model.CVbegin(mode);
         const int nl = model.nlatent();
         smurff::thread_vector<Eigen::MatrixXf> VVs(Eigen::MatrixXf::Zero(nl, nl));

         //for each column v of Vf - calculate v * vT and add to VVs
         #pragma omp parallel for schedule(guided) shared(VVs)
         for(int n = 0; n < Vf.cols(); n++) 
         {
            auto v = Vf.col(n);
            VVs.local() += v * v.transpose(); // VVs = Vvs + v * vT
         }

         VV[mode] = VVs.combine(); //accumulate sum
      }

      std::uint64_t nna() const override
      {
         return 0;
      }
   };
}
