#include "UnusedNoise.h"

#include <SmurffCpp/Utils/Error.h>

using namespace smurff;

UnusedNoise::UnusedNoise()
: INoiseModel()
{
}

float UnusedNoise::getAlpha() const 
{
   THROWERROR_NOTIMPL();
}

float UnusedNoise::sample(const SubModel& model, const PVec<> &pos, float val)
{
   THROWERROR_NOTIMPL();
}

std::ostream& UnusedNoise::info(std::ostream& os, std::string indent)
{
   os << "Noisemodel is not used here." << std::endl;
   return os;
}

std::string UnusedNoise::getStatus()
{
   return std::string("Unused");
}
