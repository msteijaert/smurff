#pragma once

#include <string>
#include <iostream>

#include "Data.h"

namespace smurff
{
   class MatrixData : public Data
   {
   public:
      int nmode() const override;
      std::ostream& info(std::ostream& os, std::string indent) override;
   };
}