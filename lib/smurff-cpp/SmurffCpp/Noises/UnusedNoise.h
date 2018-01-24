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
   double getAlpha() const override;
   double sample(const SubModel& model, const PVec<> &pos, double val) override;

   std::ostream& info(std::ostream& os, std::string indent) override;
   std::string getStatus() override;
};

}
