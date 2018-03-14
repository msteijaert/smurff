#pragma once

#include <vector>
#include <memory>
#include <string>

#include <SmurffCpp/IO/INIReader.h>

#include "MatrixConfig.h"

namespace smurff
{
   class MacauPriorConfig;

   class MacauPriorConfigItem
   {
   private:
      double m_lambda_beta;
      double m_tol;
      bool m_direct;

      std::shared_ptr<MatrixConfig> m_sideInfo; //side info matrix for macau and macauone prior

   public:
      MacauPriorConfigItem();

   public:
      std::shared_ptr<MatrixConfig> getSideInfo() const
      {
         return m_sideInfo;
      }

      void setSideInfo(std::shared_ptr<MatrixConfig> value)
      {
         m_sideInfo = value;
      }

      double getLambdaBeta() const
      {
         return m_lambda_beta;
      }

      void setLambdaBeta(double value)
      {
         m_lambda_beta = value;
      }

      double getTol() const
      {
         return m_tol;
      }

      void setTol(double value)
      {
         m_tol = value;
      }

      bool getDirect() const
      {
         return m_direct;
      }

      void setDirect(bool value)
      {
         m_direct = value;
      }

   public:
      void save(std::ofstream& os, std::size_t prior_index, std::size_t config_item_index) const;

      bool restore(const INIReader& reader, std::size_t prior_index, std::size_t config_item_index);
   };

   class MacauPriorConfig
   {
   public:
      static double LAMBDA_BETA_DEFAULT_VALUE;
      static double TOL_DEFAULT_VALUE;

   private:
      std::vector<std::shared_ptr<MacauPriorConfigItem> > m_configItems; //set of side info configs for macau and macauone priors

   public:
      MacauPriorConfig();

   public:
      void save(std::ofstream& os, std::size_t prior_index) const;

      bool restore(const INIReader& reader, std::size_t prior_index);

   public:
      const std::vector<std::shared_ptr<MacauPriorConfigItem> >& getConfigItems() const
      {
         return m_configItems;
      }

      std::vector<std::shared_ptr<MacauPriorConfigItem> >& getConfigItems()
      {
         return m_configItems;
      }
   };
}