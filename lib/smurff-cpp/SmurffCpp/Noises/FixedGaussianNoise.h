#pragma once

#include <string>
#include <iostream>

#include "INoiseModel.h"

namespace smurff {

   class NoiseFactory;

   // Gaussian noise is fixed for the whole run
   class FixedGaussianNoise : public INoiseModel
   {
      friend class NoiseFactory;
      
   public:
      double alpha;

   protected:
      FixedGaussianNoise(double a = 1.);

   public:
      double getAlpha(const SubModel& model, const PVec<> &pos, double val) override;

      std::ostream& info(std::ostream& os, std::string indent)  override;
      std::string getStatus() override;

      void setPrecision(double a);
   };

}
