#include "TensorDataFactory.h"

#include <SmurffCpp/DataTensors/TensorData.h>
#include <SmurffCpp/Noises/NoiseFactory.h>

using namespace smurff;

std::shared_ptr<Data> TensorDataFactory::create_tensor_data(std::shared_ptr<const TensorConfig> config,
                                                            const std::vector<std::shared_ptr<MatrixConfig> >& row_features,
                                                            const std::vector<std::shared_ptr<MatrixConfig> >& col_features)
{
   if (row_features.empty() && col_features.empty())
   {
      if(!config->isScarce())
         throw std::runtime_error("Tensor config should be scarse");

      std::shared_ptr<TensorData> tensorData = std::make_shared<TensorData>(*config);
      std::shared_ptr<INoiseModel> noise = NoiseFactory::create_noise_model(config->getNoiseConfig());
      tensorData->setNoiseModel(noise);
      return tensorData;
   }

   throw std::runtime_error("Tensor config does not support feature matrices");
}