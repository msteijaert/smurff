#include "NoiseConfig.h"

#include <string>
#include <set>

#include <SmurffCpp/Utils/utils.h>

using namespace smurff;

bool NoiseConfig::validate(bool throw_error) const 
{
   std::set<std::string> noise_models = { "noiseless", "fixed", "adaptive", "probit" };
   if (noise_models.find(name) == noise_models.end()) die("Unknown noise model " + name);

   return true;
}