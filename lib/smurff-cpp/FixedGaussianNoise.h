#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <memory>
#include <iomanip>
#include "utils.h"

#include "INoiseModel.h"

namespace smurff {

// Gaussian noise is fixed for the whole run
class FixedGaussianNoise : public INoiseModel
{
public:
   double alpha;

public:
   FixedGaussianNoise(double a = 1.);

   double getAlpha() override;

   std::ostream& info(std::ostream& os, std::string indent)  override;
   std::string getStatus() override;

   void setPrecision(double a);
};

}
