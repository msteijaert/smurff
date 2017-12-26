#include "TensorDataFactory.h"

#include <SmurffCpp/DataTensors/TensorData.h>
#include <SmurffCpp/Noises/NoiseFactory.h>

#include <SmurffCpp/Utils/Error.h>

using namespace smurff;

std::shared_ptr<Data> TensorDataFactory::create_tensor_data(std::shared_ptr<const TensorConfig> config,
                                                            const std::vector<std::vector<std::shared_ptr<MatrixConfig> > >& features)
{
   for (auto& fs : features)
   {
      if (!fs.empty())
      {
         THROWERROR("Tensor config does not support features");
      }
   }

   // Checking whether dense tensor is scarse makes no sense
   if(!config->isDense() && !config->isScarce())
   {
      THROWERROR("Tensor config should be scarse");
   }

   std::shared_ptr<TensorData> tensorData = std::make_shared<TensorData>(*config);
   std::shared_ptr<INoiseModel> noise = NoiseFactory::create_noise_model(config->getNoiseConfig());
   tensorData->setNoiseModel(noise);
   return tensorData;
}