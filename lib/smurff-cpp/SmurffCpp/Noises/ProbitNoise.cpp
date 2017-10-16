#include "ProbitNoise.h"

#include <assert.h>
#include <cmath>

using namespace smurff;

ProbitNoise::ProbitNoise()
   : INoiseModel()
{

}

double ProbitNoise::getAlpha()
{
   assert(false); 
   return NAN;
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