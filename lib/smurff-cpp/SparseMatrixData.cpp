#include "SparseMatrixData.h"

#if defined(_OPENMP)
#include <omp.h>
#endif

using namespace smurff;

SparseMatrixData::SparseMatrixData(Eigen::SparseMatrix<double> Y)
   : FullMatrixData<Eigen::SparseMatrix<double>>(Y)
{
   this->name = "SparseMatrixData [fully known]";
}

void SparseMatrixData::center(double global_mean)
{
    assert(Ycentered);

    IMeanCentering::center(global_mean);

    if (getCenterMode() == CenterModeTypes::CENTER_GLOBAL)
    {
      Ycentered->at(0).coeffs() -= global_mean;
      Ycentered->at(1).coeffs() -= global_mean;
    }
    else if (getCenterMode() == CenterModeTypes::CENTER_VIEW)
    {
      Ycentered->at(0).coeffs() -= getCwiseMean();
      Ycentered->at(1).coeffs() -= getCwiseMean();
    }
    else if (getCenterMode() == CenterModeTypes::CENTER_COLS)
    {
        // you cannot col/row center fully know sparse matrices
        // without converting to dense
        assert(false);
    }
    else if (getCenterMode() == CenterModeTypes::CENTER_ROWS)
    {
        // you cannot col/row center fully know sparse matrices
        // without converting to dense
        assert(false);
    }
    else if (getCenterMode() == CenterModeTypes::CENTER_NONE)
    {
       //do nothing
    }
    else
    {
       throw std::logic_error("Invalid center mode");
    }

    setCentered(true);
}

double SparseMatrixData::train_rmse(const SubModel& model) const
{
   double se = 0.;
#pragma omp parallel for schedule(guided) reduction(+:se)
   for(int c=0; c<Y.cols();++c)
   {
      int r = 0;
      for (Eigen::SparseMatrix<double>::InnerIterator it(Y, c); it; ++it)
      {
         while (r<it.row()) se += square(predict({r++,c}, model));
         se += square(it.value() - predict({r,c}, model));
      }
      for(;r<Y.rows();r++) se += square(predict({r,c}, model));
   }
   return sqrt( se / Y.rows() / Y.cols() );
}
