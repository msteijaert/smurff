#include "MatrixDataFactory.h"

#include <SmurffCpp/Configs/NoiseConfig.h>

#include <SmurffCpp/Utils/MatrixUtils.h>
#include <SmurffCpp/Utils/PVec.hpp>

//matrix classes
#include <SmurffCpp/DataMatrices/SparseMatrixData.h>
#include <SmurffCpp/DataMatrices/ScarceBinaryMatrixData.h>
#include <SmurffCpp/DataMatrices/DenseMatrixData.h>
#include <SmurffCpp/DataMatrices/MatricesData.h>

//noise classes
#include <SmurffCpp/Noises/AdaptiveGaussianNoise.h>
#include <SmurffCpp/Noises/FixedGaussianNoise.h>
#include <SmurffCpp/Noises/ProbitNoise.h>
#include <SmurffCpp/Noises/Noiseless.h>
#include <SmurffCpp/Noises/UnusedNoise.h>

using namespace smurff;

class NoiseFactory
{
public:
   static std::shared_ptr<INoiseModel> create_noise_model(const NoiseConfig& config)
   {
      if (config.name == "fixed")
      {
         return std::shared_ptr<INoiseModel>(new FixedGaussianNoise(config.precision));
      }
      else if (config.name == "adaptive")
      {
         return std::shared_ptr<INoiseModel>(new AdaptiveGaussianNoise(config.sn_init, config.sn_max));
      }
      else if (config.name == "probit")
      {
         return std::shared_ptr<INoiseModel>(new ProbitNoise());
      }
      else if (config.name == "noiseless")
      {
         return std::shared_ptr<INoiseModel>(new Noiseless());
      }
      else
      {
         throw std::runtime_error("Unknown noise model; " + config.name);
      }
   }

   static std::shared_ptr<INoiseModel> create_noise_model()
   {
      return std::shared_ptr<INoiseModel>(new UnusedNoise());
   }
};

std::shared_ptr<MatrixData> matrix_config_to_matrix(const MatrixConfig &config, bool scarce)
{
   std::shared_ptr<INoiseModel> noise = NoiseFactory::create_noise_model(config.getNoiseConfig());

   if (config.isDense())
   {
      Eigen::MatrixXd Ytrain = matrix_utils::dense_to_eigen(config);
      std::shared_ptr<MatrixData> local_data_ptr(new DenseMatrixData(Ytrain));
      local_data_ptr->setNoiseModel(noise);
      return local_data_ptr;
   }
   else
   {
      Eigen::SparseMatrix<double> Ytrain = matrix_utils::sparse_to_eigen(config);
      if (!scarce)
      {
         std::shared_ptr<MatrixData> local_data_ptr(new SparseMatrixData(Ytrain));
         local_data_ptr->setNoiseModel(noise);
         return local_data_ptr;
      }
      else if (matrix_utils::is_binary(Ytrain))
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

std::shared_ptr<MatrixData> create_matrix(const MatrixConfig &train, const std::vector<MatrixConfig> &row_features, const std::vector<MatrixConfig> &col_features)
{
   if (row_features.empty() && col_features.empty())
      return ::matrix_config_to_matrix(train, true);

   // multiple matrices
   std::shared_ptr<MatricesData> local_data_ptr(new MatricesData());
   local_data_ptr->setNoiseModel(NoiseFactory::create_noise_model());
   local_data_ptr->add(PVec<>({0,0}), ::matrix_config_to_matrix(train, true));

   for(size_t i = 0; i < row_features.size(); ++i)
   {
      local_data_ptr->add(PVec<>({0, static_cast<int>(i + 1)}), ::matrix_config_to_matrix(row_features[i], false));
   }

   for(size_t i = 0; i < col_features.size(); ++i)
   {
      local_data_ptr->add(PVec<>({static_cast<int>(i + 1), 0}), ::matrix_config_to_matrix(col_features[i], false));
   }

   return local_data_ptr;
}

std::shared_ptr<MatrixData> MatrixDataFactory::create_matrix(std::shared_ptr<Session> session)
{
   //row_matrices and col_matrices are selected if prior is not macau and not macauone
   std::vector<MatrixConfig> row_matrices;
   std::vector<MatrixConfig> col_matrices;

   if (session->config.row_prior_type != PriorTypes::macau && session->config.row_prior_type != PriorTypes::macauone)
      row_matrices = session->config.row_features;

   if (session->config.col_prior_type != PriorTypes::macau && session->config.col_prior_type != PriorTypes::macauone)
      col_matrices = session->config.col_features;

   return ::create_matrix(session->config.train, row_matrices, col_matrices);
}