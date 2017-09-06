#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <memory>
#include <iomanip>
#include "utils.h"

#include "INoiseModel.h"

namespace smurff {

class UnusedNoise : public INoiseModel 
{
public:
   UnusedNoise(Data* p);
   
   void init() override;
   void update(const SubModel& sm) override;
   double getAlpha() override;

   std::ostream& info(std::ostream& os, std::string indent) override;
   std::string getStatus() override;
};

}