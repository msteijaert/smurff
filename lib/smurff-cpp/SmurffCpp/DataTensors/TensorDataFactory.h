#pragma once

#include <memory>

#include <SmurffCpp/DataMatrices/Data.h>
#include <SmurffCpp/Configs/TensorConfig.h>

namespace smurff {
   
   class TensorDataFactory
   {
   public:
      static std::shared_ptr<Data> create_tensor_data(std::shared_ptr<TensorConfig> config);
   };
   
}