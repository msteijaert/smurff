#pragma once

#include <string>

#define NOISE_NAME_FIXED "fixed"
#define NOISE_NAME_ADAPTIVE "adaptive"
#define NOISE_NAME_PROBIT "probit"
#define NOISE_NAME_UNUSED "unused"

namespace smurff
{
   enum class NoiseTypes
   {
      fixed,
      adaptive,
      probit,
      unset,
      unused,
   };

   NoiseTypes stringToNoiseType(std::string name);
   
   std::string noiseTypeToString(NoiseTypes type);

   class NoiseConfig
   {
   public:
      //noise config
      static NoiseTypes NOISE_TYPE_DEFAULT_VALUE;
      static double PRECISION_DEFAULT_VALUE;
      static double ADAPTIVE_SN_INIT_DEFAULT_VALUE;
      static double ADAPTIVE_SN_MAX_DEFAULT_VALUE;
      static double PROBIT_DEFAULT_VALUE;

   private:
      // for fixed gaussian noise
      double m_precision;
   
      // for adaptive gausssian noise
      double m_sn_init;
      double m_sn_max;

      // for probit
      double m_threshold;

   private:
      NoiseTypes m_noise_type;

   public:
      NoiseConfig(NoiseTypes nt = NoiseTypes::unset);

   public:
      bool validate() const;

   public:
      NoiseTypes getNoiseType() const
      {
         return m_noise_type;
      }

      void setNoiseType(NoiseTypes value)
      {
         m_noise_type = value;
      }

      double getPrecision() const;

      void setPrecision(double value);

      double getSnInit() const;

      void setSnInit(double value);

      double getSnMax() const;

      void setSnMax(double value);

      double getThreshold() const;

      void setThreshold(double value);
   };
}
