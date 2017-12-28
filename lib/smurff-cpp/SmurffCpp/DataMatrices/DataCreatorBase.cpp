#include "DataCreatorBase.h"

#include <SmurffCpp/Utils/MatrixUtils.h>

//matrix classes
#include <SmurffCpp/DataMatrices/SparseMatrixData.h>
#include <SmurffCpp/DataMatrices/ScarceBinaryMatrixData.h>
#include <SmurffCpp/DataMatrices/DenseMatrixData.h>

//tensor classes
#include <SmurffCpp/DataTensors/TensorData.h>

//noise classes
#include <SmurffCpp/Configs/NoiseConfig.h>
#include <SmurffCpp/Noises/NoiseFactory.h>

using namespace smurff;

std::shared_ptr<Data> DataCreatorBase::create(std::shared_ptr<const MatrixConfig> mc) const
{
   std::shared_ptr<INoiseModel> noise = NoiseFactory::create_noise_model(mc->getNoiseConfig());

   if (mc->isDense())
   {
      Eigen::MatrixXd Ytrain = matrix_utils::dense_to_eigen(*mc);
      std::shared_ptr<MatrixData> local_data_ptr(new DenseMatrixData(Ytrain));
      local_data_ptr->setNoiseModel(noise);
      return local_data_ptr;
   }
   else
   {
      Eigen::SparseMatrix<double> Ytrain = matrix_utils::sparse_to_eigen(*mc);
      if (!mc->isScarce())
      {
         std::shared_ptr<MatrixData> local_data_ptr(new SparseMatrixData(Ytrain));
         local_data_ptr->setNoiseModel(noise);
         return local_data_ptr;
      }
      else if (matrix_utils::is_explicit_binary(Ytrain))
      {
         std::shared_ptr<MatrixData> local_data_ptr(new ScarceBinaryMatrixData(Ytrain));
         local_data_ptr->setNoiseModel(noise);
         return local_data_ptr;
      }
      else
      {
         std::shared_ptr<MatrixData> local_data_ptr(new ScarceMatrixData(Ytrain));
         local_data_ptr->setNoiseModel(noise);
         return local_data_ptr;
      }
   }
}

std::shared_ptr<Data> DataCreatorBase::create(std::shared_ptr<const TensorConfig> tc) const
{
   // Checking whether dense tensor is scarse makes no sense
   if(!tc->isDense() && !tc->isScarce())
   {
      THROWERROR("Tensor config should be scarse");
   }

   std::shared_ptr<TensorData> tensorData = std::make_shared<TensorData>(*tc);
   std::shared_ptr<INoiseModel> noise = NoiseFactory::create_noise_model(tc->getNoiseConfig());
   tensorData->setNoiseModel(noise);
   return tensorData;
}