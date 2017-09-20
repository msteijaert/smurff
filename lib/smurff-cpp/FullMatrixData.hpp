#pragma once

#if defined(_OPENMP)
#include <omp.h>
#endif

#include "utils.h"

namespace smurff
{
   template<class YType>
   class FullMatrixData : public MatrixDataTempl<YType>
   {
   private:
      Eigen::MatrixXd VV[2];

   public:
      FullMatrixData(YType Y) : MatrixDataTempl<YType>(Y)
      {
         this->name = "MatrixData [fully known]";
      }


   public:
      void get_pnm(const SubModel& model, int mode, int d, Eigen::VectorXd& rr, Eigen::MatrixXd& MM) override
      {
         const double alpha = this->noise().getAlpha();
         auto &Y = this->Yc.at(mode);
         rr.noalias() += (model.V(mode) * Y.col(d)) * alpha;
         MM.noalias() += VV[mode] * alpha;
      }

      void update_pnm(const SubModel& model, int mode) override
      {
          auto &Vf = model.V(mode);
          const int nl = model.nlatent();
          thread_vector<Eigen::MatrixXd> VVs(Eigen::MatrixXd::Zero(nl, nl));

      #pragma omp parallel for schedule(dynamic, 8) shared(VVs)
          for(int n = 0; n < Vf.cols(); n++) {
              auto v = Vf.col(n);
              VVs.local() += v * v.transpose();
          }

          VV[mode] = VVs.combine();
      }

      int nna() const override
      {
         return 0;
      }

   private:
      double compute_mode_mean(int m, int c) override
      {
          const auto &col = this->Yc.at(m).col(c);
          if (col.nonZeros() == 0) return this->cwise_mean;
          return col.sum() / this->Yc.at(m).rows();
      }
   };
}