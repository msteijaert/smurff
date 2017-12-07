#pragma once

#include "MatrixDataTempl.hpp"

#include <SmurffCpp/VMatrixExprIterator.hpp>
#include <SmurffCpp/ConstVMatrixExprIterator.hpp>

namespace smurff
{
   template<class YType>
   class FullMatrixData : public MatrixDataTempl<YType>
   {
   private:
      Eigen::MatrixXd VV[2]; // sum of v * vT, where v is column of V

   public:
      FullMatrixData(YType Y) 
         : MatrixDataTempl<YType>(Y)
      {
         this->name = "MatrixData [fully known]";
      }

   public:
      //d is an index of column in U matrix
      void get_pnm(const SubModel& model, uint32_t mode, int d, Eigen::VectorXd& rr, Eigen::MatrixXd& MM) override
      {
         auto& Y = this->Y(mode);
          auto Vf = *model.CVbegin(mode);
          for(int r = 0; r<Y.rows(); ++r)
          {
              // FIXME!
              int pos0 = r;
              int pos1 = d;
              if (mode == 0) std::swap(pos0, pos1);
              // FIXME!!!!
              const double alpha = this->noise()->getAlpha(model.predict({r,d}),0.0);
              rr.noalias() += (Vf.col(r) * Y.col(d)) * alpha; // rr = rr + (V[m] * y[d]) * alpha
          }
          MM.noalias() += VV[mode]; // MM = MM + VV[m] * alpha
      }

      //purpose of update_pnm is to cache VV matrix
      void update_pnm(const SubModel& model, uint32_t mode) override
      {
         auto Vf = *model.CVbegin(mode);
         const int nl = model.nlatent();
         thread_vector<Eigen::MatrixXd> VVs(Eigen::MatrixXd::Zero(nl, nl));

         //for each column v of Vf - calculate v * vT and add to VVs
         #pragma omp parallel for schedule(dynamic, 8) shared(VVs)
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
