#pragma once

#include <memory>
#include <vector>

#include <SmurffCpp/DataMatrices/Data.h>
#include <SmurffCpp/Configs/MatrixConfig.h>

namespace smurff {
   
   class TensorDataFactory
   {
   public:
      static std::shared_ptr<Data> create_tensor_data(std::shared_ptr<const TensorConfig> config,
                                                      const std::vector<std::shared_ptr<MatrixConfig> >& row_features, 
                                                      const std::vector<std::shared_ptr<MatrixConfig> >& col_features);
   };
   
}