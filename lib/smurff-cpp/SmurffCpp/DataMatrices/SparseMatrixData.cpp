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
   for(int c = 0; c < Y().cols();++c)
   {
      int r = 0;
      for (Eigen::SparseMatrix<double>::InnerIterator it(Y(), c); it; ++it)
      {
         while (r<it.row()) se += square(predict({r++,c}, model));
         se += square(it.value() - predict({r,c}, model));
      }
      for(;r < Y().rows();r++) se += square(predict({r,c}, model));
   }
   return sqrt( se / Y().rows() / Y().cols() );
}
