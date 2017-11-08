#pragma once

#include "MatrixDataTempl.hpp"

namespace smurff
{
   template<class YType>
   class FullMatrixData : public MatrixDataTempl<YType>
   {
   private:
      Eigen::MatrixXd VV[2];

   public:
      FullMatrixData(YType Y) 
         : MatrixDataTempl<YType>(Y)
      {
         this->name = "MatrixData [fully known]";
      }

   public:
      void get_pnm(const SubModel& model, int mode, int d, Eigen::VectorXd& rr, Eigen::MatrixXd& MM) override
      {
         const double alpha = this->noise()->getAlpha();
         auto& Y = this->Y(mode);
         rr.noalias() += (model.V(mode) * Y.col(d)) * alpha; // rr = rr + (V[m] * y[d]) * alpha
         MM.noalias() += VV[mode] * alpha; // MM = MM + VV[m] * alpha
      }

      void update_pnm(const SubModel& model, int mode) override
      {
         auto &Vf = model.V(mode);
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

      int nna() const override
      {
         return 0;
      }
   };
}