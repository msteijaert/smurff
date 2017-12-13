#include "ProbitNoise.h"

#include <SmurffCpp/Utils/Error.h>

using namespace smurff;

ProbitNoise::ProbitNoise()
   : INoiseModel()
{

}

double ProbitNoise::getAlpha()
{
   THROWERROR_NOTIMPL();
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