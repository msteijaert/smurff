#pragma once

#include "Data.h"

namespace smurff
{
   class MatrixData : public Data
   {
   public:
      int nmode() const override;
      std::ostream& info(std::ostream& os, std::string indent) override;
      int nrow() const;
      int ncol() const;
   };
}
