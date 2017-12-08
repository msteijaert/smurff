#pragma once

#include <string>
#include <iostream>
#include <memory>

#include <SmurffCpp/Configs/TensorConfig.h>

namespace smurff { namespace generic_io 
{
   std::shared_ptr<TensorConfig> read_data_config(const std::string& filename, bool isScarce);

   void write_data_config(const std::string& filename, std::shared_ptr<TensorConfig> tensorConfig);

   bool file_exists(const std::string& filepath);
}}