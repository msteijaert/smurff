#pragma once

#include <string>
#include <iostream>

#include "INoiseModel.h"

namespace smurff {

class Noiseless : public INoiseModel
{
public:
   Noiseless();

   double getAlpha() override;

   std::ostream& info(std::ostream& os, std::string indent) override;
   std::string getStatus() override;
};

}
