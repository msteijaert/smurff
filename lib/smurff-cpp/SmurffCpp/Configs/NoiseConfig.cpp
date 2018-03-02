#include "NoiseConfig.h"

#include <string>
#include <set>

#include <SmurffCpp/Utils/Error.h>

using namespace smurff;

#define NOISE_NAME_FIXED "fixed"
#define NOISE_NAME_ADAPTIVE "adaptive"
#define NOISE_NAME_PROBIT "probit"
#define NOISE_NAME_UNUSED "unused"
#define NOISE_NAME_UNSET "unset"

//noise config
NoiseTypes NoiseConfig::NOISE_TYPE_DEFAULT_VALUE = NoiseTypes::fixed;
double NoiseConfig::PRECISION_DEFAULT_VALUE = 5.0;
double NoiseConfig::ADAPTIVE_SN_INIT_DEFAULT_VALUE = 1.0;
double NoiseConfig::ADAPTIVE_SN_MAX_DEFAULT_VALUE = 10.0;
double NoiseConfig::PROBIT_DEFAULT_VALUE = 0.0;

NoiseConfig::NoiseConfig(NoiseTypes nt)
   : m_noise_type(nt)
{
   m_precision = PRECISION_DEFAULT_VALUE;

   // for adaptive gausssian noise
   m_sn_init = ADAPTIVE_SN_INIT_DEFAULT_VALUE;
   m_sn_max = ADAPTIVE_SN_MAX_DEFAULT_VALUE;

   // for probit
   m_threshold = PROBIT_DEFAULT_VALUE;
}

NoiseTypes smurff::stringToNoiseType(std::string name)
{
   if(name == NOISE_NAME_FIXED)
      return NoiseTypes::fixed;
   else if(name == NOISE_NAME_ADAPTIVE)
      return NoiseTypes::adaptive;
   else if(name == NOISE_NAME_PROBIT)
      return NoiseTypes::probit;
   else if(name == NOISE_NAME_UNUSED)
      return NoiseTypes::unused;
   else if (name == NOISE_NAME_UNSET)
      return NoiseTypes::unset;
   else
   {
      THROWERROR("Invalid noise type " + name);
   }
}

std::string smurff::noiseTypeToString(NoiseTypes type)
{
   switch(type)
   {
      case NoiseTypes::fixed:
         return NOISE_NAME_FIXED;
      case NoiseTypes::adaptive:
         return NOISE_NAME_ADAPTIVE;
      case NoiseTypes::probit:
         return NOISE_NAME_PROBIT;
      case NoiseTypes::unused:
         return NOISE_NAME_UNUSED;
      case NoiseTypes::unset:
         return NOISE_NAME_UNSET;
      default:
      {
         THROWERROR("Invalid noise type");
      }
   }
}

bool NoiseConfig::validate() const 
{
   return true;
}

double NoiseConfig::getPrecision() const
{
   return m_precision;
}

void NoiseConfig::setPrecision(double value)
{
   m_precision = value;
}

double NoiseConfig::getSnInit() const
{
   return m_sn_init;
}

void NoiseConfig::setSnInit(double value)
{
   m_sn_init = value;
}

double NoiseConfig::getSnMax() const
{
   return m_sn_max;
}

void NoiseConfig::setSnMax(double value)
{
   m_sn_max = value;
}

double NoiseConfig::getThreshold() const
{
   return m_threshold;
}

void NoiseConfig::setThreshold(double value)
{
   m_threshold = value;
}