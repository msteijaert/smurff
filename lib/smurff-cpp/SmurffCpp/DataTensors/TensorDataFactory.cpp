#include "TensorDataFactory.h"

#include <SmurffCpp/DataTensors/TensorData.h>

using namespace smurff;

std::shared_ptr<Data> TensorDataFactory::create_tensor_data(std::shared_ptr<const TensorConfig> config,
                                                            const std::vector<std::shared_ptr<MatrixConfig> >& row_features, 
                                                            const std::vector<std::shared_ptr<MatrixConfig> >& col_features)
{
   if (row_features.empty() && col_features.empty())
      return std::make_shared<TensorData>(*config);

   throw std::runtime_error("Tensor config does not support feature matrices");
}