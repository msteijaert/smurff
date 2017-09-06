#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <memory>
#include <iomanip>
#include "utils.h"

#include "INoiseModel.h"

namespace smurff {
   
   // Gaussian noise that adapts to the data
   class ProbitNoise : public INoiseModel 
   {
      public:
         ProbitNoise(Data* p);

         void init() override;
         void update(const SubModel& sm) override;
         double getAlpha() override;

         std::ostream& info(std::ostream& os, std::string indent) override;
         std::string getStatus() override;
   };
   
}