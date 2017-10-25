#include "DenseMatrixData.h"

using namespace smurff;

DenseMatrixData::DenseMatrixData(Eigen::MatrixXd Y)
   : FullMatrixData<Eigen::MatrixXd>(Y)
{
    this->name = "DenseMatrixData [fully known]";
}

void DenseMatrixData::center(double global_mean)
{
    IMeanCentering::center(global_mean);

    if (getCenterMode() == CenterModeTypes::CENTER_GLOBAL)
    {
      getYcPtr()->at(0).array() -= global_mean;
      getYcPtr()->at(1).array() -= global_mean;
    }
    else if (getCenterMode() == CenterModeTypes::CENTER_VIEW)
    {
      getYcPtr()->at(0).array() -= getCwiseMean();
      getYcPtr()->at(1).array() -= getCwiseMean();
    }
    else if (getCenterMode() == CenterModeTypes::CENTER_COLS)
    {
      getYcPtr()->at(0).rowwise() -= getModeMean(0).transpose();
      getYcPtr()->at(1) = getYcPtr()->at(0).transpose();
    }
    else if (getCenterMode() == CenterModeTypes::CENTER_ROWS)
    {
      getYcPtr()->at(1).rowwise() -= getModeMean(1).transpose();
      getYcPtr()->at(0) = getYcPtr()->at(1).transpose();
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
