#pragma once

#include <string>
#include <iostream>

#include "INoiseModel.h"

#include <DataMatrices/Data.h>

namespace smurff {

// Gaussian noise that adapts to the data
class AdaptiveGaussianNoise : public INoiseModel
{
public:
   double var_total = NAN;
   double alpha = NAN;
   double alpha_max = NAN;
   double sn_max;
   double sn_init;

public:
   AdaptiveGaussianNoise(double sinit = 1., double smax = 10.);

   void init(const Data* data) override;
   void update(const Data* data, const SubModel& model) override;
   double getAlpha() override;

   std::ostream &info(std::ostream &os, std::string indent) override;
   std::string getStatus() override;

   void setSNInit(double a);
   void setSNMax(double a);
};

}