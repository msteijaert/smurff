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
         while (r < it.row()) //handle implicit zeroes
            se += square(model.predict({r++, c}));

         se += square(it.value() - model.predict({r, c}));
      }

      for(; r < Y().rows(); r++) //handle implicit zeroes
         se += square(model.predict({r, c}));
   }
   return sqrt(se / Y().rows() / Y().cols());
}

double SparseMatrixData::var_total() const
{
   throw std::runtime_error("not implemented");
}

double SparseMatrixData::sumsq(const SubModel& model) const
{
   throw std::runtime_error("not implemented");
}