#include "DataCreator.h"

#include "MatrixDataFactory.h"
#include <SmurffCpp/DataTensors/TensorDataFactory.h>

#include <SmurffCpp/Utils/Error.h>

using namespace smurff;

std::shared_ptr<Data> DataCreator::create(std::shared_ptr<const MatrixConfig> mc) const
{
   //row_matrices and col_matrices are selected if prior is not macau and not macauone
   std::vector<std::shared_ptr<MatrixConfig> > row_matrices;
   std::vector<std::shared_ptr<MatrixConfig> > col_matrices;

   if (m_session->config.getRowPriorType() != PriorTypes::macau && m_session->config.getRowPriorType() != PriorTypes::macauone)
      row_matrices = m_session->config.getRowFeatures();

   if (m_session->config.getColPriorType() != PriorTypes::macau && m_session->config.getColPriorType() != PriorTypes::macauone)
      col_matrices = m_session->config.getColFeatures();

   return MatrixDataFactory::create_matrix_data(mc, row_matrices, col_matrices);
}

std::shared_ptr<Data> DataCreator::create(std::shared_ptr<const TensorConfig> tc) const
{
   //row_matrices and col_matrices are selected if prior is not macau and not macauone
   std::vector<std::shared_ptr<MatrixConfig> > row_matrices;
   std::vector<std::shared_ptr<MatrixConfig> > col_matrices;

   if (m_session->config.getRowPriorType() != PriorTypes::normal && m_session->config.getRowPriorType() != PriorTypes::default_prior)
   {
      THROWERROR("Currently only normal prior is supported");
   }

   if (m_session->config.getColPriorType() != PriorTypes::normal && m_session->config.getColPriorType() != PriorTypes::default_prior)
   {
      THROWERROR("Currently only normal prior is supported");
   }

   if (m_session->config.getRowPriorType() != PriorTypes::macau && m_session->config.getRowPriorType() != PriorTypes::macauone)
      row_matrices = m_session->config.getRowFeatures();

   if (m_session->config.getColPriorType() != PriorTypes::macau && m_session->config.getColPriorType() != PriorTypes::macauone)
      col_matrices = m_session->config.getColFeatures();

   return TensorDataFactory::create_tensor_data(tc, row_matrices, col_matrices);
}