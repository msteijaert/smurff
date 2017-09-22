#include "DenseMatrixData.h"

#if defined(_OPENMP)
#include <omp.h>
#endif

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
        Yc.at(0).array() -= global_mean;
        Yc.at(1).array() -= global_mean;
    }
    else if (getCenterMode() == CenterModeTypes::CENTER_VIEW)
    {
        Yc.at(0).array() -= getCwiseMean();
        Yc.at(1).array() -= getCwiseMean();
    }
    else if (getCenterMode() == CenterModeTypes::CENTER_COLS)
    {
        Yc.at(0).rowwise() -= getModeMean(0).transpose();
        Yc.at(1) = Yc.at(0).transpose();
    }
    else if (getCenterMode() == CenterModeTypes::CENTER_ROWS)
    {
        Yc.at(1).rowwise() -= getModeMean(1).transpose();
        Yc.at(0) = Yc.at(1).transpose();
    }
    else
    {
       throw std::logic_error("Invalid center mode");
    }
}

double DenseMatrixData::train_rmse(const SubModel& model) const
{
    double se = 0.;
#pragma omp parallel for schedule(guided) reduction(+:se)
    for(int c=0; c<Y.cols();++c) {
        for(int m=0; m<Y.rows(); ++m) {
            se += square(Y(m,c) - predict({m,c}, model));
        }
    }
    return sqrt( se / Y.rows() / Y.cols() );
}
