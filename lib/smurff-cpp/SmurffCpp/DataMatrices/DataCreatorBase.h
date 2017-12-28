#pragma once

#include <memory>

#include "IDataCreator.h"

namespace smurff {
   class DataCreatorBase : public IDataCreator
   {
   public:
      DataCreatorBase()
      {
      }

   public:
      std::shared_ptr<Data> create(std::shared_ptr<const MatrixConfig> mc) const override;
      std::shared_ptr<Data> create(std::shared_ptr<const TensorConfig> tc) const override;
   };
}