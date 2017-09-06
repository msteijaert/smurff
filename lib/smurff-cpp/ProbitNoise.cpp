#include "ProbitNoise.h"

using namespace Eigen;
using namespace smurff;

ProbitNoise::ProbitNoise(Data* p) 
   : INoiseModel(p) 
{
   
}

void ProbitNoise::init() 
{
   
}

void ProbitNoise::update(const SubModel& sm)
{

}

double ProbitNoise::getAlpha()
{
   assert(false); return NAN; 
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