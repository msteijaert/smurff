#include "SparseMatrixData.h"

using namespace smurff;

SparseMatrixData::SparseMatrixData(Eigen::SparseMatrix<double> Y)
   : FullMatrixData<Eigen::SparseMatrix<double>>(Y)
{
   this->name = "SparseMatrixData [fully known]";
}

double SparseMatrixData::train_rmse(const SubModel& model) const
{
   double se = 0.;
   #pragma omp parallel for schedule(guided) reduction(+:se)
   for(int c = 0; c < Y().cols(); ++c)
   {
      int r = 0;
      for (Eigen::SparseMatrix<double>::InnerIterator it(Y(), c); it; ++it)
      {
         for(; r < it.row(); r++) //handle implicit zeroes
            se += square(model.predict({r, c}));

         se += square(it.value() - model.predict({r, c}));
         r++;
      }

      for(; r < Y().rows(); r++) //handle implicit zeroes
         se += square(model.predict({r, c}));
   }
   return sqrt(se / Y().rows() / Y().cols());
}

double SparseMatrixData::var_total() const
{
   double cwise_mean = this->sum() / (this->size() - this->nna());
   double se = 0.0;

   #pragma omp parallel for schedule(dynamic, 4) reduction(+:se)
   for(int c = 0; c < Y().cols(); ++c)
   {
      int r = 0;
      for (Eigen::SparseMatrix<double>::InnerIterator it(Y(), c); it; ++it)
      {
         for(; r < it.row(); r++) //handle implicit zeroes
            se += square(cwise_mean);

         se += square(it.value() - cwise_mean);
         r++;
      }

      for(; r < Y().rows(); r++) //handle implicit zeroes
         se += square(cwise_mean);
   }

   double var = se / nnz();
   if (var <= 0.0 || std::isnan(var))
   {
      // if var cannot be computed using 1.0
      var = 1.0;
   }

   return var;
}

double SparseMatrixData::sumsq(const SubModel& model) const
{
   double sumsq = 0.0;
   
   #pragma omp parallel for schedule(dynamic, 4) reduction(+:sumsq)
   for(int c = 0; c < Y().cols(); ++c)
   {
      int r = 0;
      for (Eigen::SparseMatrix<double>::InnerIterator it(Y(), c); it; ++it)
      {
         for(; r < it.row(); r++) //handle implicit zeroes
            sumsq += square(model.predict({r, c}));

         sumsq += square(model.predict({r, c}) - it.value());
         r++;
      }

      for(; r < Y().rows(); r++) //handle implicit zeroes
         sumsq += square(model.predict({r, c}));
   }

   return sumsq;
}