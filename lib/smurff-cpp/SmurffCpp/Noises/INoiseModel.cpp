#include "INoiseModel.h"

#include <SmurffCpp/DataMatrices/Data.h>

using namespace smurff;

float INoiseModel::getAlpha() const
{
    return 1.0;
}

float INoiseModel::sample(const SubModel& model, const PVec<> &pos, float val)
{
    return getAlpha() * val;
}
