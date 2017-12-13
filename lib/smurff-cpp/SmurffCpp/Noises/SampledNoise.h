#pragma once

#include <string>
#include <iostream>

#include "INoiseModel.h"

namespace smurff {

   // Gaussian noise is fixed for the whole run
   class SampledNoise : public INoiseModel
   {
   public:
      void getMuLambda(const SubModel& model, int mode, int d, Eigen::VectorXd& rr, Eigen::MatrixXd& MM) override;
      virtual double sample(const SubModel& model, const PVec<> &pos, double val) = 0;
   };

}
