#pragma once

#include <string>
#include <iostream>

#include "SampledNoise.h"

namespace smurff {

   class NoiseFactory;

   // Gaussian noise that adapts to the data
   class ProbitNoise : public SampledNoise
   {
      friend class NoiseFactory;
      
   protected:
      ProbitNoise();

   public:
      double sample(const SubModel& model, const PVec<> &pos, double val) override;

      std::ostream& info(std::ostream& os, std::string indent) override;
      std::string getStatus() override;
   };

}
