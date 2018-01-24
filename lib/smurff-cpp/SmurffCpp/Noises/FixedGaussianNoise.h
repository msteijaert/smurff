#pragma once

#include <string>
#include <iostream>

#include "GaussianNoise.h"

namespace smurff {

   class NoiseFactory;

   // Gaussian noise is fixed for the whole run
   class FixedGaussianNoise : public GaussianNoise
   {
      friend class NoiseFactory;

   protected:
      FixedGaussianNoise(double a = 1.);

   public:
      std::ostream& info(std::ostream& os, std::string indent)  override;
      std::string getStatus() override;

      void setPrecision(double a);
   };

}
