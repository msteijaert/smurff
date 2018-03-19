#include "PriorFactory.h"

#include <Eigen/Core>

#include <SmurffCpp/Priors/NormalPrior.h>
#include <SmurffCpp/Priors/NormalOnePrior.h>
#include <SmurffCpp/Priors/SpikeAndSlabPrior.h>

#include <SmurffCpp/Utils/Error.h>

#include <SmurffCpp/SideInfo/DenseDoubleFeat.h>
#include <SmurffCpp/SideInfo/SparseDoubleFeat.h>
#include <SmurffCpp/SideInfo/SparseFeat.h>

#include <SmurffCpp/Utils/MatrixUtils.h>

using namespace smurff;
using namespace Eigen;

//create macau prior features

std::shared_ptr<ISideInfo> PriorFactory::side_info_config_to_dense_features(std::shared_ptr<MatrixConfig> sideinfoConfig, int mode)
{
   Eigen::MatrixXd sideinfo = matrix_utils::dense_to_eigen(*sideinfoConfig);
   auto side_info_ptr = std::make_shared<Eigen::MatrixXd>(sideinfo);
   return std::make_shared<DenseDoubleFeatSideInfo>(side_info_ptr);
}

std::shared_ptr<ISideInfo> PriorFactory::side_info_config_to_sparse_binary_features(std::shared_ptr<MatrixConfig> sideinfoConfig, int mode)
{
   std::uint64_t nrow = sideinfoConfig->getNRow();
   std::uint64_t ncol = sideinfoConfig->getNCol();
   std::uint64_t nnz = sideinfoConfig->getNNZ();

   std::shared_ptr<std::vector<std::uint32_t> > rows = sideinfoConfig->getRowsPtr();
   std::shared_ptr<std::vector<std::uint32_t> > cols = sideinfoConfig->getColsPtr();

   // Temporary solution. As soon as SparseFeat works with vectors instead of pointers,
   // we will remove these extra memory allocation and manipulation
   int* rowsRawPtr = new int[nnz];
   int* colsRawPtr = new int[nnz];
   for (std::uint64_t i = 0; i < nnz; i++)
   {
      rowsRawPtr[i] = rows->operator[](i);
      colsRawPtr[i] = cols->operator[](i);
   }

   auto side_info_ptr = std::make_shared<SparseFeat>(nrow, ncol, nnz, rowsRawPtr, colsRawPtr);
   return std::make_shared<SparseFeatSideInfo>(side_info_ptr);
}

std::shared_ptr<ISideInfo> PriorFactory::side_info_config_to_sparse_features(std::shared_ptr<MatrixConfig> sideinfoConfig, int mode)
{
   std::uint64_t nrow = sideinfoConfig->getNRow();
   std::uint64_t ncol = sideinfoConfig->getNCol();
   std::uint64_t nnz = sideinfoConfig->getNNZ();

   std::shared_ptr<std::vector<std::uint32_t> > rows = sideinfoConfig->getRowsPtr();
   std::shared_ptr<std::vector<std::uint32_t> > cols = sideinfoConfig->getColsPtr();
   std::shared_ptr<std::vector<double> > values = sideinfoConfig->getValuesPtr();

   // Temporary solution. As soon as SparseDoubleFeat works with vectorsor shared pointers instead of raw pointers,
   // we will remove these extra memory allocation and manipulation
   int* rowsRawPtr = new int[nnz];
   int* colsRawPtr = new int[nnz];
   double* valuesRawPtr = new double[nnz];
   for (size_t i = 0; i < nnz; i++)
   {
      rowsRawPtr[i] = rows->operator[](i);
      colsRawPtr[i] = cols->operator[](i);
      valuesRawPtr[i] = values->operator[](i);
   }

   auto side_info_ptr = std::make_shared<SparseDoubleFeat>(nrow, ncol, nnz, rowsRawPtr, colsRawPtr, valuesRawPtr);
   return std::make_shared<SparseDoubleFeatSideInfo>(side_info_ptr);
}

//-------

std::shared_ptr<ILatentPrior> PriorFactory::create_macau_prior(std::shared_ptr<Session> session, PriorTypes prior_type,
   const std::vector<std::shared_ptr<ISideInfo> >& side_infos,
   const std::vector<std::shared_ptr<MacauPriorConfigItem> >& config_items)
{
   if(prior_type == PriorTypes::macau || prior_type == PriorTypes::default_prior)
   {
      return create_macau_prior<MacauPrior>(session, side_infos, config_items);
   }
   else if(prior_type == PriorTypes::macauone)
   {
      return create_macau_prior<MacauOnePrior>(session, side_infos, config_items);
   }
   else
   {
      THROWERROR("Unknown prior with side info: " + priorTypeToString(prior_type));
   }
}

std::shared_ptr<ILatentPrior> PriorFactory::create_prior(std::shared_ptr<Session> session, int mode)
{
   PriorTypes priorType = session->getConfig().getPriorTypes().at(mode);

   switch(priorType)
   {
   case PriorTypes::normal:
   case PriorTypes::default_prior:
      return std::shared_ptr<NormalPrior>(new NormalPrior(session, -1));
   case PriorTypes::spikeandslab:
      return std::shared_ptr<SpikeAndSlabPrior>(new SpikeAndSlabPrior(session, -1));
   case PriorTypes::normalone:
      return std::shared_ptr<NormalOnePrior>(new NormalOnePrior(session, -1));
   case PriorTypes::macau:
   case PriorTypes::macauone:
      return create_macau_prior<PriorFactory>(session, mode, priorType, session->getConfig().getMacauPriorConfigs().at(mode));
   default:
      {
         THROWERROR("Unknown prior: " + priorTypeToString(priorType));
      }
   }
}
