#pragma once

#include <memory>

#include <SmurffCpp/Configs/NoiseConfig.h>
#include <SmurffCpp/Noises/INoiseModel.h>

namespace smurff {

   class NoiseFactory
   {
   public:
      static std::shared_ptr<INoiseModel> create_noise_model(const NoiseConfig& config);
   };
}