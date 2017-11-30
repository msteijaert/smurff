#include "MatrixDataTempl.hpp"

namespace smurff
{
   template<>
   double MatrixDataTempl<Eigen::MatrixXd>::var_total() const
   {
      double cwise_mean = this->sum() / (this->size() - this->nna());
      double se = (Y().array() - cwise_mean).square().sum();
      double var = se / Y().nonZeros();
      if (var <= 0.0 || std::isnan(var))
      {
         // if var cannot be computed using 1.0
         var = 1.0;
      }

      return var;
   }

   template<>
   double MatrixDataTempl<Eigen::SparseMatrix<double> >::var_total() const
   {
      double cwise_mean = this->sum() / (this->size() - this->nna());
      double se = 0.0;

      #pragma omp parallel for schedule(dynamic, 4) reduction(+:se)
      for (int k = 0; k < Y().outerSize(); ++k)
      {
         for (Eigen::SparseMatrix<double>::InnerIterator it(Y(), k); it; ++it)
         {
            se += square(it.value() - cwise_mean);
         }
      }

      double var = se / Y().nonZeros();
      if (var <= 0.0 || std::isnan(var))
      {
         // if var cannot be computed using 1.0
         var = 1.0;
      }

      return var;
   }

   // for the adaptive gaussian noise
   template<>
   double MatrixDataTempl<Eigen::MatrixXd>::sumsq(const SubModel& model) const
   {
      double sumsq = 0.0;

      #pragma omp parallel for schedule(dynamic, 4) reduction(+:sumsq)
      for (int j = 0; j < this->ncol(); j++) 
      {
         for (int i = 0; i < this->nrow(); i++) 
         {
               sumsq += square(predict({i,j}, model) - this->Y()(i,j));
         }
      }

      return sumsq;
   }

   template<>
   double MatrixDataTempl<Eigen::SparseMatrix<double> >::sumsq(const SubModel& model) const
   {
      double sumsq = 0.0;

      #pragma omp parallel for schedule(dynamic, 4) reduction(+:sumsq)
      for (int j = 0; j < Y().outerSize(); j++) 
      {
         for (Eigen::SparseMatrix<double>::InnerIterator it(Y(), j); it; ++it) 
         {
               int i = it.row();
               sumsq += square(predict({i,j}, model)- it.value());
         }
      }

      return sumsq;
   }
}
