#include "FixedGaussianNoise.h"

using namespace Eigen;
using namespace smurff;

FixedGaussianNoise::FixedGaussianNoise(Data* p, double a) 
   : INoiseModel(p), alpha(a)  
{

}

void FixedGaussianNoise::init() 
{

}

void FixedGaussianNoise::update(const SubModel& sm)
{

}

double FixedGaussianNoise::getAlpha()
{
   return alpha; 
}

std::ostream& FixedGaussianNoise::info(std::ostream& os, std::string indent)
{ 
   os << "Fixed gaussian noise with precision: " << alpha << "\n";
   return os;
}

std::string FixedGaussianNoise::getStatus()
{
   return std::string("Fixed: ") + std::to_string(alpha); 
}

void FixedGaussianNoise::setPrecision(double a)
{
   alpha = a; 
}