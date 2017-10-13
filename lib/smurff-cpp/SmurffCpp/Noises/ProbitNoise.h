#pragma once

#include <string>
#include <iostream>

#include "INoiseModel.h"

namespace smurff {

   // Gaussian noise that adapts to the data
   class ProbitNoise : public INoiseModel
   {
      public:
         ProbitNoise();

         double getAlpha() override;

         std::ostream& info(std::ostream& os, std::string indent) override;
         std::string getStatus() override;
   };

}