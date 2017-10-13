#include "UnusedNoise.h"

#include <assert.h>
#include <cmath>

using namespace smurff;

UnusedNoise::UnusedNoise()
: INoiseModel()
{
}

double UnusedNoise::getAlpha()
{
   assert(false);
   return NAN;
}

std::ostream& UnusedNoise::info(std::ostream& os, std::string indent)
{
   os << "Noisemodel is not used here.\n";
   return os;
}

std::string UnusedNoise::getStatus()
{
   return std::string("Unused");
}