#pragma once

// _OPENMP will be enabled if -fopenmp flag is passed to the compiler (use cmake release build)
#if defined(_OPENMP)
#include <omp.h>
#endif

#include "MatrixDataTempl.hpp"

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
         auto &Y = this->getYcPtr()->at(mode);
         rr.noalias() += (model.V(mode) * Y.col(d)) * alpha;
         MM.noalias() += VV[mode] * alpha;
      }

      void update_pnm(const SubModel& model, int mode) override
      {
         auto &Vf = model.V(mode);
         const int nl = model.nlatent();
         thread_vector<Eigen::MatrixXd> VVs(Eigen::MatrixXd::Zero(nl, nl));

         #pragma omp parallel for schedule(dynamic, 8) shared(VVs)
         for(int n = 0; n < Vf.cols(); n++) 
         {
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
      double compute_mode_mean_mn(int mode, int pos) override
      {
          const auto &col = this->getYcPtr()->at(mode).col(pos);
          if (col.nonZeros() == 0)
            return this->getCwiseMean();
          return col.sum() / this->getYcPtr()->at(mode).rows();
      }
   };
}