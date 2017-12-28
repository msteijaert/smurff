#include "PriorFactory.h"

#include <Eigen/Core>

#include <SmurffCpp/Priors/MacauOnePrior.hpp>
#include <SmurffCpp/Priors/MacauPrior.hpp>
#include <SmurffCpp/Priors/NormalPrior.h>
#include <SmurffCpp/Priors/SpikeAndSlabPrior.h>

#include <SmurffCpp/Utils/Error.h>

using namespace smurff;
using namespace Eigen;

//create macau prior features

std::shared_ptr<Eigen::MatrixXd> side_info_config_to_dense_features(std::shared_ptr<MatrixConfig> sideinfoConfig, int mode)
{
   Eigen::MatrixXd sideinfo = matrix_utils::dense_to_eigen(*sideinfoConfig);
   return std::shared_ptr<Eigen::MatrixXd>(new Eigen::MatrixXd(sideinfo));
}

std::shared_ptr<SparseFeat> side_info_config_to_sparse_binary_features(std::shared_ptr<MatrixConfig> sideinfoConfig, int mode)
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

   return std::shared_ptr<SparseFeat>(new SparseFeat(nrow, ncol, nnz, rowsRawPtr, colsRawPtr));
}

std::shared_ptr<SparseDoubleFeat> side_info_config_to_sparse_features(std::shared_ptr<MatrixConfig> sideinfoConfig, int mode)
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

   return std::shared_ptr<SparseDoubleFeat>(new SparseDoubleFeat(nrow, ncol, nnz, rowsRawPtr, colsRawPtr, valuesRawPtr));
}

//-------

template<class SideInfo>
std::shared_ptr<ILatentPrior> create_macau_prior(std::shared_ptr<Session> session, PriorTypes prior_type, std::shared_ptr<SideInfo> side_info)
{
   if(prior_type == PriorTypes::macau || prior_type == PriorTypes::default_prior)
   {
      std::shared_ptr<MacauPrior<SideInfo>> prior(new MacauPrior<SideInfo>(session, -1));

      prior->addSideInfo(side_info, session->config.getDirect());
      prior->setLambdaBeta(session->config.getLambdaBeta());
      prior->setTol(session->config.getTol());

      return prior;
   }
   else if(prior_type == PriorTypes::macauone)
   {
      std::shared_ptr<MacauOnePrior<SideInfo>> prior(new MacauOnePrior<SideInfo>(session, -1));

      prior->addSideInfo(side_info, session->config.getDirect());
      prior->setLambdaBeta(session->config.getLambdaBeta());

      return prior;
   }
   else
   {
      THROWERROR("Unknown prior with side info: " + priorTypeToString(prior_type));
   }
}

//mode - 0 (row), 1 (col)
//vsideinfo - vector of side feature configs (row or col)
std::shared_ptr<ILatentPrior> create_macau_prior(std::shared_ptr<Session> session, int mode, PriorTypes prior_type, const std::shared_ptr<MatrixConfig>& sideinfoConfig)
{
   if(!sideinfoConfig)
   {
      THROWERROR("Only one feature matrix is allowed");
   }

   if (sideinfoConfig->isBinary())
   {
      std::shared_ptr<SparseFeat> sideinfo = side_info_config_to_sparse_binary_features(sideinfoConfig, mode);
      return create_macau_prior(session, prior_type, sideinfo);
   }
   else if (sideinfoConfig->isDense())
   {
      std::shared_ptr<Eigen::MatrixXd> sideinfo = side_info_config_to_dense_features(sideinfoConfig, mode);
      return create_macau_prior(session, prior_type, sideinfo);
   }
   else
   {
      std::shared_ptr<SparseDoubleFeat> sideinfo = side_info_config_to_sparse_features(sideinfoConfig, mode);
      return create_macau_prior(session, prior_type, sideinfo);
   }
}

//-------

std::shared_ptr<ILatentPrior> PriorFactory::create_prior(std::shared_ptr<Session> session, int mode)
{
   PriorTypes priorType = session->config.getPriorTypes().at(mode);

   switch(priorType)
   {
   case PriorTypes::normal:
   case PriorTypes::default_prior:
      return std::shared_ptr<NormalPrior>(new NormalPrior(session, -1));
   case PriorTypes::spikeandslab:
      return std::shared_ptr<SpikeAndSlabPrior>(new SpikeAndSlabPrior(session, -1));
   case PriorTypes::macau:
   case PriorTypes::macauone:
      return create_macau_prior(session, mode, priorType, session->config.getSideInfo().at(mode));
   default:
      {
         THROWERROR("Unknown prior: " + priorTypeToString(priorType));
      }
   }
}