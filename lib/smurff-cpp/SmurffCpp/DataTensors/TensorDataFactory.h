#pragma once

#include <memory>
#include <vector>

#include <SmurffCpp/DataMatrices/Data.h>

namespace smurff {
   
   class TensorDataFactory
   {
   public:
      static std::shared_ptr<Data> create_tensor_data(std::shared_ptr<const TensorConfig> config,
                                                      const std::vector<std::vector<std::shared_ptr<MatrixConfig> > >& features);
   };
   
}
