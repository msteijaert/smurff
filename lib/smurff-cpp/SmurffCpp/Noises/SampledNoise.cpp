#include "SampledNoise.h"

#include <SmurffCpp/Utils/Distribution.h>
#include <SmurffCpp/DataMatrices/Data.h>

using namespace smurff;

void SampledNoise::getMuLambda(const SubModel& model, int mode, int d, Eigen::VectorXd& rr, Eigen::MatrixXd& MM)
{
    data().getMuLambda(model, mode, d, *this, rr, MM);
}
