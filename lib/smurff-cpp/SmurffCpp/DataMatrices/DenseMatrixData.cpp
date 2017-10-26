#include "DenseMatrixData.h"

using namespace smurff;

DenseMatrixData::DenseMatrixData(Eigen::MatrixXd Y)
   : FullMatrixData<Eigen::MatrixXd>(Y)
{
    this->name = "DenseMatrixData [fully known]";
}

double DenseMatrixData::train_rmse(const SubModel& model) const
{
   double se = 0.;
   #pragma omp parallel for schedule(guided) reduction(+:se)
   for(int c=0; c<Y.cols();++c) 
   {
      for(int m=0; m<Y.rows(); ++m) 
      {
         se += square(Y(m,c) - predict({m,c}, model));
      }
   }
   return sqrt( se / Y.rows() / Y.cols() );
}
