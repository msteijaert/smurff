#pragma once

#include <string>
#include <iostream>

#include "INoiseModel.h"

namespace smurff {

class NoiseFactory;

class UnusedNoise : public INoiseModel
{
   friend class NoiseFactory;

protected:
   UnusedNoise();

public:
   void getMuLambda(const SubModel& model, int mode, int d, Eigen::VectorXd& rr, Eigen::MatrixXd& MM) override;

   std::ostream& info(std::ostream& os, std::string indent) override;
   std::string getStatus() override;
};

}
