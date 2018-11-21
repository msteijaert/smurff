#pragma once

#include <string>
#include <iostream>

#include <SmurffCpp/Noises/GaussianNoise.h>

#include <SmurffCpp/DataMatrices/Data.h>

namespace smurff {

   class NoiseFactory;

   // Gaussian noise that adapts to the data
   class AdaptiveGaussianNoise : public GaussianNoise
   {
      friend class NoiseFactory;
      
   public:
      float var_total = NAN;
      float alpha_max = NAN;
      float sn_max;
      float sn_init;

   protected:
      AdaptiveGaussianNoise(float sinit = 1., float smax = 10.);

   public:
      void init(const Data* data) override;
      void update(const SubModel& model) override;

      std::ostream &info(std::ostream &os, std::string indent) override;
      std::string getStatus() override;

      void setSNInit(float a);
      void setSNMax(float a);
   };

}
