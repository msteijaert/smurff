#include "AdaptiveGaussianNoise.h"

#include <SmurffCpp/Utils/Distribution.h>
#include <SmurffCpp/DataMatrices/Data.h>

using namespace smurff;

void GaussianNoise::getMuLambda(const SubModel& model, int mode, int d, Eigen::VectorXd& rr, Eigen::MatrixXd& MM)
{
    INoiseModel::getMuLambda(model, mode, d, rr, MM);
    
    rr *= alpha;
    MM *= alpha;
}
