#include "DataFactory.h"

#include <SmurffCpp/Utils/MatrixUtils.h>
#include <SmurffCpp/Utils/PVec.hpp>

#include <SmurffCpp/Configs/TensorConfig.h>

using namespace smurff;

std::shared_ptr<Data> MatrixDataFactory::create_matrix(std::shared_ptr<Session> session)
{
   //row_matrices and col_matrices are selected if prior is not macau and not macauone
   std::vector<std::shared_ptr<TensorConfig> > row_matrices;
   std::vector<std::shared_ptr<TensorConfig> > col_matrices;

   if (session->config.row_prior_type != PriorTypes::macau && session->config.row_prior_type != PriorTypes::macauone)
      std::copy(session->config.m_row_features.begin(), session->config.m_row_features.end(), row_matrices.end());

   if (session->config.col_prior_type != PriorTypes::macau && session->config.col_prior_type != PriorTypes::macauone)
      std::copy(session->config.m_col_features.begin(), session->config.m_col_features.end(), col_matrices.end());

   return session->config.m_train->toData(row_matrices, col_matrices);
}