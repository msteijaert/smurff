#include "AdaptiveGaussianNoise.h"

#include <SmurffCpp/Utils/Distribution.h>
#include <SmurffCpp/DataMatrices/Data.h>

using namespace smurff;

double GaussianNoise::getAlpha() const
{
    return alpha;
}
