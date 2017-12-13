#include "INoiseModel.h"

#include <SmurffCpp/DataMatrices/Data.h>

using namespace smurff;

void INoiseModel::getMuLambda(const SubModel& model, int mode, int d, Eigen::VectorXd& rr, Eigen::MatrixXd& MM)
{
    data().getMuLambda(model, mode, d, rr, MM);
}

double INoiseModel::sample(const SubModel& model, const PVec<> &pos, double val)
{
    return val;
}
