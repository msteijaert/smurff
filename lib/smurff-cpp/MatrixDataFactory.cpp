#include "MatrixDataFactory.h"

#include "MatrixUtils.h"
#include "PVec.h"

#include "NoiseConfig.h"

//matrix classes
#include "SparseMatrixData.h"
#include "ScarceBinaryMatrixData.h"
#include "DenseMatrixData.h"
#include "MatricesData.h"

//noise classes
#include "AdaptiveGaussianNoise.h"
#include "FixedGaussianNoise.h"
#include "ProbitNoise.h"
#include "Noiseless.h"
#include "UnusedNoise.h"

using namespace smurff;

void setNoiseModel(const NoiseConfig &config, Data* data)
{
    if (config.name == "fixed")
    {
        data->setNoiseModel(new FixedGaussianNoise(config.precision));
    }
    else if (config.name == "adaptive")
    {
        data->setNoiseModel(new AdaptiveGaussianNoise(config.sn_init, config.sn_max));
    }
    else if (config.name == "probit")
    {
        data->setNoiseModel(new ProbitNoise());
    }
    else if (config.name == "noiseless")
    {
        data->setNoiseModel(new Noiseless());
    }
    else
    {
        die("Unknown noise model; " + config.name);
    }
}

std::unique_ptr<MatrixData> matrix_config_to_matrix(const MatrixConfig &config, bool scarce)
{
   std::unique_ptr<MatrixData> local_data_ptr;

   if (config.isDense()) 
   {
      Eigen::MatrixXd Ytrain = dense_to_eigen(config);
      local_data_ptr = std::unique_ptr<MatrixData>(new DenseMatrixData(Ytrain));
   } 
   else 
   {
      SparseMatrixD Ytrain = sparse_to_eigen(config);
      if (!scarce) 
      {
         local_data_ptr = std::unique_ptr<MatrixData>(new SparseMatrixData(Ytrain));
      } 
      else if (is_binary(Ytrain)) 
      {
         local_data_ptr = std::unique_ptr<MatrixData>(new ScarceBinaryMatrixData(Ytrain));
      } 
      else 
      {
         local_data_ptr = std::unique_ptr<MatrixData>(new ScarceMatrixData(Ytrain));
      }
   }

   setNoiseModel(config.getNoiseConfig(), local_data_ptr.get());

   return local_data_ptr;
}

std::unique_ptr<MatrixData> smurff::matrix_config_to_matrix(const MatrixConfig &train, 
                                                            const std::vector<MatrixConfig> &row_features, 
                                                            const std::vector<MatrixConfig> &col_features)
{
   if (row_features.empty() && col_features.empty()) 
   {
      return ::matrix_config_to_matrix(train, true);
   }

   // multiple matrices
   MatricesData* local_data_ptr = new MatricesData();

   local_data_ptr->setNoiseModel(new UnusedNoise());
   local_data_ptr->add(PVec({0,0}), ::matrix_config_to_matrix(train, true));

   for(size_t i = 0; i < row_features.size(); ++i) 
   {
      local_data_ptr->add(PVec({0, static_cast<int>(i + 1)}), ::matrix_config_to_matrix(row_features[i], false));
   }

   for(size_t i = 0; i < col_features.size(); ++i) 
   {
      local_data_ptr->add(PVec({static_cast<int>(i + 1), 0}), ::matrix_config_to_matrix(col_features[i], false));
   }

   return std::unique_ptr<MatrixData>(local_data_ptr);
}