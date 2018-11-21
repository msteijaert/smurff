#pragma once

#include <string>
#include <iostream>

#include "FixedGaussianNoise.h"

namespace smurff {

   class NoiseFactory;

   // Gaussian noise, noise value is sampled around fixed precision
   // used in MacauPrior
   class SampledGaussianNoise : public FixedGaussianNoise
   {
      friend class NoiseFactory;

   protected:
      SampledGaussianNoise(float a = 1.);

   public:
      std::ostream& info(std::ostream& os, std::string indent)  override;
      std::string getStatus() override;
   };

}
