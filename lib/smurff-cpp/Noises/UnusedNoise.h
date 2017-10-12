#pragma once

#include <string>
#include <iostream>

#include "INoiseModel.h"

namespace smurff {

class UnusedNoise : public INoiseModel
{
public:
   UnusedNoise();

   double getAlpha() override;

   std::ostream& info(std::ostream& os, std::string indent) override;
   std::string getStatus() override;
};

}