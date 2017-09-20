#include "MatrixDataTempl.hpp"

namespace smurff
{
   template<>
   double MatrixDataTempl<Eigen::MatrixXd>::var_total() const
   {
      auto &Y = Yc.at(0);
      double se = Y.array().square().sum();
      double var = se / Y.nonZeros();
      if (var <= 0.0 || std::isnan(var)) {
         // if var cannot be computed using 1.0
         var = 1.0;
      }

      return var;
   }

   template<>
   double MatrixDataTempl<Eigen::SparseMatrix<double> >::var_total() const
   {
      double se = 0.0;
      auto &Y = Yc.at(0);

   #pragma omp parallel for schedule(dynamic, 4) reduction(+:se)
      for (int k = 0; k < Y.outerSize(); ++k) {
         for (Eigen::SparseMatrix<double>::InnerIterator it(Y,k); it; ++it) {
               se += square(it.value());
         }
      }

      double var = se / Y.nonZeros();
      if (var <= 0.0 || std::isnan(var)) {
         // if var cannot be computed using 1.0
         var = 1.0;
      }

      return var;
   }

   //macau
   /*
   double var_total(MatrixData & matrixData)
   {
   double se = 0.0;
   double mean_value = matrixData.mean_value;

   #pragma omp parallel for schedule(dynamic, 4) reduction(+:se)
   for (int k = 0; k < matrixData.Y.outerSize(); ++k) {
      for (SparseMatrix<double>::InnerIterator it(matrixData.Y, k); it; ++it) {
         se += square(it.value() - mean_value);
      }
   }

   var_total = se / matrixData.Y.nonZeros();
   if (var_total <= 0.0 || std::isnan(var_total)) {
      // if var cannot be computed using 1.0
      var_total = 1.0;
   }

   return var_total;
   }
   */

   //macau
   /*
   double var_total(TensorData & data)
   {
   double se = 0.0;
   double mean_value = data.mean_value;

   auto& sparseMode   = (*data.Y)[0];
   VectorXd & values  = sparseMode->values;

      #pragma omp parallel for schedule(dynamic, 4) reduction(+:se)
   for (int i = 0; i < values.size(); i++) {
      se += square(values(i) - mean_value);
   }
   var_total = se / values.size();
   if (var_total <= 0.0 || std::isnan(var_total)) {
      var_total = 1.0;
   }

   return var_total;
   }
   */

   // for the adaptive gaussian noise
   template<>
   double MatrixDataTempl<Eigen::MatrixXd>::sumsq(const SubModel &model) const
   {
      double sumsq = 0.0;

   #pragma omp parallel for schedule(dynamic, 4) reduction(+:sumsq)
      for (int j = 0; j < this->ncol(); j++) {
         for (int i = 0; i < this->nrow(); i++) {
               double Yhat = model.dot({j,i}) + offset_to_mean({j,i});
               sumsq += square(Yhat - this->Y(i,j));
         }
      }

      return sumsq;
   }

   template<>
   double MatrixDataTempl<Eigen::SparseMatrix<double> >::sumsq(const SubModel &model) const
   {
      double sumsq = 0.0;

   #pragma omp parallel for schedule(dynamic, 4) reduction(+:sumsq)
      for (int j = 0; j < Y.outerSize(); j++) {
         for (Eigen::SparseMatrix<double>::InnerIterator it(Y, j); it; ++it) {
               int i = it.row();
               double Yhat = model.dot({j,i}) + offset_to_mean({j,i});
               sumsq += square(Yhat - it.value());
         }
      }

      return sumsq;
   }

   //macau
   /*
   double sumsq(MatrixData & data, std::vector< std::unique_ptr<Eigen::MatrixXd> > & samples)
   {
   double sumsq = 0.0;
   MatrixXd & U = *samples[0];
   MatrixXd & V = *samples[1];

   Eigen::SparseMatrix<double> & train = data.Y;
   double mean_value = data.mean_value;

   #pragma omp parallel for schedule(dynamic, 4) reduction(+:sumsq)
   for (int j = 0; j < train.outerSize(); j++) {
      auto Vj = V.col(j);
      for (SparseMatrix<double>::InnerIterator it(train, j); it; ++it) {
         double Yhat = Vj.dot( U.col(it.row()) ) + mean_value;
         sumsq += square(Yhat - it.value());
      }
   }

   return sumsq;
   }
   */

   //macau
   /*
   double sumsq(TensorData & data, std::vector< std::unique_ptr<Eigen::MatrixXd> > & samples)
   {
   double sumsq = 0.0;
   double mean_value = data.mean_value;

   auto& sparseMode = (*data.Y)[0];
   auto& U = samples[0];

   const int nmodes = samples.size();
   const int num_latents = U->rows();

   #pragma omp parallel for schedule(dynamic, 4) reduction(+:sumsq)
   for (int n = 0; n < data.dims(0); n++) {
      Eigen::VectorXd u = U->col(n);
      for (int j = sparseMode->row_ptr(n);
               j < sparseMode->row_ptr(n + 1);
               j++)
      {
         VectorXi idx = sparseMode->indices.row(j);
         // computing prediction from tensor
         double Yhat = mean_value;
         for (int d = 0; d < num_latents; d++) {
         double tmp = u(d);

         for (int m = 1; m < nmodes; m++) {
            tmp *= (*samples[m])(d, idx(m - 1));
         }
         Yhat += tmp;
         }
         sumsq += square(Yhat - sparseMode->values(j));
      }

   }

      return sumsq;
   }
   */
}