#include "UnusedNoise.h"

using namespace Eigen;
using namespace smurff;

UnusedNoise::UnusedNoise(Data* p)
: INoiseModel(p)
{
}

void UnusedNoise::init()
{
}

void UnusedNoise::update(const SubModel& sm)
{
}

double UnusedNoise::getAlpha()
{
   assert(false); 
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