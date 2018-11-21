#include "FixedGaussianNoise.h"

#include <iomanip>

using namespace smurff;

FixedGaussianNoise::FixedGaussianNoise(float a)
   : GaussianNoise()
{
    alpha = a;
}

std::ostream& FixedGaussianNoise::info(std::ostream& os, std::string indent)
{
   os << "Fixed gaussian noise with precision: " << std::fixed << std::setprecision(2) << alpha << std::endl;
   return os;
}

std::string FixedGaussianNoise::getStatus()
{
   return std::string("Fixed: ") + std::to_string(alpha);
}

void FixedGaussianNoise::setPrecision(float a)
{
   alpha = a;
}
