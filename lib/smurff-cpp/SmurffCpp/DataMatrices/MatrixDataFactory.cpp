#include "MatrixDataFactory.h"

#include <SmurffCpp/Utils/MatrixUtils.h>
#include <SmurffCpp/Utils/PVec.hpp>

#include <SmurffCpp/Configs/TensorConfig.h>

//matrix classes
#include <SmurffCpp/DataMatrices/SparseMatrixData.h>
#include <SmurffCpp/DataMatrices/ScarceMatrixData.h>
#include <SmurffCpp/DataMatrices/DenseMatrixData.h>
#include <SmurffCpp/DataMatrices/MatricesData.h>

//noise classes
#include <SmurffCpp/Configs/NoiseConfig.h>
#include <SmurffCpp/Noises/NoiseFactory.h>

using namespace smurff;

std::shared_ptr<MatrixData> create_matrix_data(std::shared_ptr<const MatrixConfig> matrixConfig)
{
   std::shared_ptr<INoiseModel> noise = NoiseFactory::create_noise_model(matrixConfig->getNoiseConfig());
   
   if (matrixConfig->isDense())
   {
      Eigen::MatrixXf Ytrain = matrix_utils::dense_to_eigen(*matrixConfig);
      std::shared_ptr<MatrixData> local_data_ptr(new DenseMatrixData(Ytrain));
      local_data_ptr->setNoiseModel(noise);
      return local_data_ptr;
   }
   else
   {
      Eigen::SparseMatrix<float> Ytrain = matrix_utils::sparse_to_eigen(*matrixConfig);
      if (!matrixConfig->isScarce())
      {
         std::shared_ptr<MatrixData> local_data_ptr(new SparseMatrixData(Ytrain));
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

std::shared_ptr<Data> MatrixDataFactory::create_matrix_data(std::shared_ptr<const MatrixConfig> matrixConfig, 
                                                            const std::vector<std::shared_ptr<MatrixConfig> >& row_features, 
                                                            const std::vector<std::shared_ptr<MatrixConfig> >& col_features)
{
   if (row_features.empty() && col_features.empty())
      return ::create_matrix_data(matrixConfig);

   // multiple matrices
   NoiseConfig ncfg(NoiseTypes::unused);
   std::shared_ptr<MatricesData> local_data_ptr(new MatricesData());
   local_data_ptr->setNoiseModel(NoiseFactory::create_noise_model(ncfg));
   local_data_ptr->add(PVec<>({0,0}), ::create_matrix_data(matrixConfig));

   for(size_t i = 0; i < row_features.size(); ++i)
   {
      local_data_ptr->add(PVec<>({0, static_cast<int>(i + 1)}), ::create_matrix_data(row_features[i]));
   }

   for(size_t i = 0; i < col_features.size(); ++i)
   {
      local_data_ptr->add(PVec<>({static_cast<int>(i + 1), 0}), ::create_matrix_data(col_features[i]));
   }

   return local_data_ptr;
}
