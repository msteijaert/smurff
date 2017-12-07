#include "ProbitNoise.h"

#include <assert.h>
#include <cmath>

#include <SmurffCpp/Utils/TruncNorm.h>

using namespace smurff;

ProbitNoise::ProbitNoise()
   : INoiseModel()
{

}

double ProbitNoise::getAlpha(double pred, double val)
{
    return rand_truncnorm(pred * val, 1.0, 0.0);
}

std::ostream& ProbitNoise::info(std::ostream& os, std::string indent)
{
   os << "Probit Noise\n";
   return os;
}

std::string ProbitNoise::getStatus()
{
   return std::string();
}
