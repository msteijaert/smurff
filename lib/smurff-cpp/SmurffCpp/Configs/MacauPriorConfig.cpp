#include <iostream>
#include <fstream>
#include <memory>

#include "MacauPriorConfig.h"

#include "TensorConfig.h"

#define MACAU_PRIOR_CONFIG_PREFIX_TAG "macau_prior_config"
#define MACAU_PRIOR_CONFIG_ITEM_PREFIX_TAG "macau_prior_config_item"

#define NUM_SIDE_INFO_TAG "num_side_info"

#define TOL_TAG "tol"
#define DIRECT_TAG "direct"
#define SIDE_INFO_PREFIX "side_info"

using namespace smurff;

double MacauPriorConfig::BETA_PRECISION_DEFAULT_VALUE = 10.0;
double MacauPriorConfig::TOL_DEFAULT_VALUE = 1e-6;

MacauPriorConfigItem::MacauPriorConfigItem()
{
   m_tol = MacauPriorConfig::TOL_DEFAULT_VALUE;
   m_direct = false;
}

void MacauPriorConfigItem::save(std::ofstream& os, std::size_t prior_index, std::size_t config_item_index) const
{
   //macau prior config item section
   os << "[" << MACAU_PRIOR_CONFIG_ITEM_PREFIX_TAG << "_" << prior_index << "_" << config_item_index << "]" << std::endl;

   //config item data
   os << TOL_TAG << " = " << m_tol << std::endl;
   os << DIRECT_TAG << " = " << m_direct << std::endl;

   os << std::endl;

   std::stringstream ss;
   ss << SIDE_INFO_PREFIX << "_" << prior_index;

   //config item side info
   TensorConfig::save_tensor_config(os, ss.str(), config_item_index, m_sideInfo);

   os << std::endl;
}

bool MacauPriorConfigItem::restore(const INIFile& reader, std::size_t prior_index, std::size_t config_item_index)
{
   auto add_index = [](const std::string name, int idx = -1) -> std::string
   {
      if (idx >= 0)
         return name + "_" + std::to_string(idx);
      return name;
   };

   std::stringstream section;
   section << MACAU_PRIOR_CONFIG_ITEM_PREFIX_TAG << "_" << prior_index << "_" << config_item_index;

   //restore side info properties
   m_tol = reader.getReal(section.str(), TOL_TAG, MacauPriorConfig::TOL_DEFAULT_VALUE);
   m_direct = reader.getBoolean(section.str(), DIRECT_TAG, false);

   std::stringstream ss;
   ss << SIDE_INFO_PREFIX << "_" << prior_index;

   auto tensor_cfg = TensorConfig::restore_tensor_config(reader, add_index(ss.str(), config_item_index));
   m_sideInfo = std::dynamic_pointer_cast<MatrixConfig>(tensor_cfg);

   return true;
}

MacauPriorConfig::MacauPriorConfig()
{
   
}

void MacauPriorConfig::save(std::ofstream& os, std::size_t prior_index) const
{
   //macau prior config section
   os << "[" << MACAU_PRIOR_CONFIG_PREFIX_TAG << "_" << prior_index << "]" << std::endl;

   //number of side infos
   os << NUM_SIDE_INFO_TAG << " = " << m_configItems.size() << std::endl;

   os << std::endl;

   //write side info section
   for (std::size_t config_item_index = 0; config_item_index < m_configItems.size(); config_item_index++)
   {
      auto& ci = m_configItems.at(config_item_index);
      THROWERROR_ASSERT(ci);

      ci->save(os, prior_index, config_item_index);
   }
}

bool MacauPriorConfig::restore(const INIFile& reader, std::size_t prior_index)
{
   std::stringstream section;
   section << MACAU_PRIOR_CONFIG_PREFIX_TAG << "_" << prior_index;

   size_t num_side_info = reader.getInteger(section.str(), NUM_SIDE_INFO_TAG, 0);

   if (num_side_info == 0)
      return false;

   for (std::size_t config_item_index = 0; config_item_index < num_side_info; config_item_index++)
   {
      auto configItem = std::make_shared<MacauPriorConfigItem>();
      configItem->restore(reader, prior_index, config_item_index);
      m_configItems.push_back(configItem);
   }

   return true;
}