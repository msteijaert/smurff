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
    this->global_mean = global_mean;
    if (center_mode == CENTER_GLOBAL) {
        Yc.at(0).array() -= global_mean;
        Yc.at(1).array() -= global_mean;
    } else if (center_mode == CENTER_VIEW) {
        Yc.at(0).array() -= cwise_mean;
        Yc.at(1).array() -= cwise_mean;
    } else if (center_mode == CENTER_COLS) {
        Yc.at(0).rowwise() -= mode_mean.at(0).transpose();
        Yc.at(1) = Yc.at(0).transpose();
    } else if (center_mode == CENTER_ROWS) {
        Yc.at(1).rowwise() -= mode_mean.at(1).transpose();
        Yc.at(0) = Yc.at(1).transpose();
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
