#include "NoiseFactory.h"

#include <SmurffCpp/Noises/AdaptiveGaussianNoise.h>
#include <SmurffCpp/Noises/FixedGaussianNoise.h>
#include <SmurffCpp/Noises/ProbitNoise.h>
#include <SmurffCpp/Noises/Noiseless.h>
#include <SmurffCpp/Noises/UnusedNoise.h>

#include <SmurffCpp/Utils/Error.h>

using namespace smurff;

std::shared_ptr<INoiseModel> NoiseFactory::create_noise_model(const NoiseConfig& config)
{
   switch(config.getNoiseType())
   {
      case NoiseTypes::fixed:
         return std::shared_ptr<INoiseModel>(new FixedGaussianNoise(config.precision));
      case NoiseTypes::adaptive:
         return std::shared_ptr<INoiseModel>(new AdaptiveGaussianNoise(config.sn_init, config.sn_max));
      case NoiseTypes::probit:
         return std::shared_ptr<INoiseModel>(new ProbitNoise());
      case NoiseTypes::noiseless:
         return std::shared_ptr<INoiseModel>(new Noiseless());
      case NoiseTypes::unused:
         return std::shared_ptr<INoiseModel>(new UnusedNoise());
      default:
         THROWERROR("Unknown noise model type");
   }
}