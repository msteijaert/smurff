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

   //row prior
   PriorTypes rowPriorType = m_session->config.getPriorTypes().at(0);
   if (rowPriorType != PriorTypes::macau && rowPriorType != PriorTypes::macauone)
      row_matrices = m_session->config.getFeatures().at(0);

   //col prior
   PriorTypes colPriorType = m_session->config.getPriorTypes().at(1);
   if (colPriorType != PriorTypes::macau && colPriorType != PriorTypes::macauone)
      col_matrices = m_session->config.getFeatures().at(1);

   return MatrixDataFactory::create_matrix_data(mc, row_matrices, col_matrices);
}

std::shared_ptr<Data> DataCreator::create(std::shared_ptr<const TensorConfig> tc) const
{
   std::vector<std::vector<std::shared_ptr<MatrixConfig> > > features;

   for(std::size_t i = 0; i < m_session->config.getPriorTypes().size(); i++)
   {
      PriorTypes pt = m_session->config.getPriorTypes()[i];

      if (pt != PriorTypes::macau && pt != PriorTypes::macauone)
         features.push_back(m_session->config.getFeatures().at(i));
      else
         features.push_back(std::vector<std::shared_ptr<MatrixConfig> >());
   }

   return TensorDataFactory::create_tensor_data(tc, features);
}