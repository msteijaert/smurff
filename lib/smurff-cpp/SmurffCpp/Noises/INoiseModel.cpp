#include "INoiseModel.h"

#include <SmurffCpp/DataMatrices/Data.h>

using namespace smurff;

double INoiseModel::getAlpha() const
{
    return 1.0;
}

double INoiseModel::sample(const SubModel& model, const PVec<> &pos, double val)
{
    return getAlpha() * val;
}
