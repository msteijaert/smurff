#include "SampledGaussianNoise.h"

#include <iomanip>

using namespace smurff;

SampledGaussianNoise::SampledGaussianNoise(double a)
   : FixedGaussianNoise(a)
{
}

std::ostream& SampledGaussianNoise::info(std::ostream& os, std::string indent)
{
   os << "Sampled gaussian noise with precision: " << std::fixed << std::setprecision(2) << alpha << std::endl;
   return os;
}

std::string SampledGaussianNoise::getStatus()
{
   return std::string("Sampled with fixed precision: ") + std::to_string(alpha);
}