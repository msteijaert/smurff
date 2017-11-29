#pragma once

#include <string>
#include <iostream>
#include <memory>

#include <SmurffCpp/Configs/TensorConfig.h>

namespace smurff { namespace generic_io 
{
   std::shared_ptr<TensorConfig> read_data_config(const std::string& filename);

   void write_data_config(const std::string& filename, std::shared_ptr<TensorConfig> tensorConfig);
}}