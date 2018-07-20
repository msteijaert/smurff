#pragma once

#include <string>

namespace smurff
{
   enum class NoiseTypes
   {
      fixed,
      sampled,
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

      static std::string get_default_string()
      {
         return smurff::noiseTypeToString(NoiseConfig::NOISE_TYPE_DEFAULT_VALUE) +
            "," + std::to_string(NoiseConfig::PRECISION_DEFAULT_VALUE) +
            "," + std::to_string(NoiseConfig::ADAPTIVE_SN_INIT_DEFAULT_VALUE) +
            "," + std::to_string(NoiseConfig::ADAPTIVE_SN_MAX_DEFAULT_VALUE) +
            "," + std::to_string(NoiseConfig::PROBIT_DEFAULT_VALUE);
      }

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

      std::string getNoiseTypeAsString() const;

      void setNoiseType(std::string);

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
