#pragma once

#include <string>
#include <iostream>

#include "INoiseModel.h"

namespace smurff {

   // Gaussian noise is fixed for the whole run
   class GaussianNoise : public INoiseModel
   {
   public:
      double alpha = NAN;

      void getMuLambda(const SubModel& model, int mode, int d, Eigen::VectorXd& rr, Eigen::MatrixXd& MM) override;
   };

}
