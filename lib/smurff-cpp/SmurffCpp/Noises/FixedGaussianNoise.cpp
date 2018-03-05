#include "FixedGaussianNoise.h"

using namespace smurff;

FixedGaussianNoise::FixedGaussianNoise(double a)
   : GaussianNoise()
{
    alpha = a;
}

std::ostream& FixedGaussianNoise::info(std::ostream& os, std::string indent)
{
   os << "Fixed gaussian noise with precision: " << alpha << std::endl;
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
