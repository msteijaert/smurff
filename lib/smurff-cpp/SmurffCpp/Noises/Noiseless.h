#pragma once

#include <string>
#include <iostream>

#include "INoiseModel.h"

namespace smurff {

   class NoiseFactory;

   class Noiseless : public INoiseModel
   {
      friend class NoiseFactory;
      
   protected:
      Noiseless();

   public:
      double getAlpha() override;

      std::ostream& info(std::ostream& os, std::string indent) override;
      std::string getStatus() override;
   };

}
