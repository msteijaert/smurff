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
   
   // Temporary solution #2
   // macau expects the rows of the matrix to be equal to the mode size, 
   // if the mode == 1 (col_features) we need to swap the rows and columns
   if (mode == 1) 
      sideinfo.transposeInPlace();

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

   // Temporary solution #2
   // macau expects the rows of the matrix to be equal to the mode size, 
   // if the mode == 1 (col_features) we need to swap the rows and columns
   if (mode == 1) 
   {
       std::swap(nrow, ncol);
       std::swap(rowsRawPtr, colsRawPtr);
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

   // Temporary solution #2
   // macau expects the rows of the matrix to be equal to the mode
   // size, if the mode == 1 (col_features) we need to swap the rows and columns
   if (mode == 1) 
   {
       std::swap(nrow, ncol);
       std::swap(rowsRawPtr, colsRawPtr);
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
      
      prior->addSideInfo(side_info, session->config.direct);
      prior->setLambdaBeta(session->config.lambda_beta);
      prior->setTol(session->config.tol);

      return prior;
   }
   else if(prior_type == PriorTypes::macauone)
   {
      std::shared_ptr<MacauOnePrior<SideInfo>> prior(new MacauOnePrior<SideInfo>(session, -1));
      
      prior->addSideInfo(side_info, session->config.direct);
      prior->setLambdaBeta(session->config.lambda_beta);

      return prior;
   }
   else
   {
      THROWERROR("Unknown prior with side info: " + priorTypeToString(prior_type));
   }
}

//mode - 0 (row), 1 (col)
//vsideinfo - vector of side feature configs (row or col)
std::shared_ptr<ILatentPrior> create_macau_prior(std::shared_ptr<Session> session, int mode, PriorTypes prior_type, const std::vector<std::shared_ptr<MatrixConfig> >& vsideinfo)
{
   if(vsideinfo.size() != 1)
   {
      THROWERROR("Only one feature matrix is allowed");
   }

   std::shared_ptr<MatrixConfig> sideinfoConfig = vsideinfo.at(0);

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

//mode - 0 (row), 1 (col)
//vsideinfo - vector of side feature configs (row or col)
std::shared_ptr<ILatentPrior> create_prior(std::shared_ptr<Session> session, int mode, PriorTypes prior_type, const std::vector<std::shared_ptr<MatrixConfig> >& vsideinfo)
{
   // row prior with side information
   // side information can only be applied to macau and macauone priors
   if (vsideinfo.size())
   {
      switch(prior_type)
      {
      case PriorTypes::macau:
      case PriorTypes::macauone:
         return create_macau_prior(session, mode, prior_type, vsideinfo);
      default:
         {
            THROWERROR("SideInfo only with macau(one) prior");
         }
      }
   }
   else
   {
      switch(prior_type)
      {
      case PriorTypes::normal:
      case PriorTypes::default_prior:
         return std::shared_ptr<NormalPrior>(new NormalPrior(session, -1));
      case PriorTypes::spikeandslab:
         return std::shared_ptr<SpikeAndSlabPrior>(new SpikeAndSlabPrior(session, -1));
      default:
         {
            THROWERROR("Unknown prior without side info: " + priorTypeToString(prior_type));
         }
      }
   }
}

std::shared_ptr<ILatentPrior> PriorFactory::create_prior(std::shared_ptr<Session> session, int mode)
{
   switch(mode)
   {
      case 0:
      {
         //row_sideinfo and col_sideinfo are selected if prior is macau or macauone
         std::vector<std::shared_ptr<MatrixConfig> > row_sideinfo;

         if (session->config.row_prior_type == PriorTypes::macau || session->config.row_prior_type == PriorTypes::macauone)
            row_sideinfo = session->config.m_row_features;

         return ::create_prior(session, mode, session->config.row_prior_type, row_sideinfo);
      }
      case 1:
      {
         //row_sideinfo and col_sideinfo are selected if prior is macau or macauone
         std::vector<std::shared_ptr<MatrixConfig> > col_sideinfo;

         if (session->config.col_prior_type == PriorTypes::macau || session->config.col_prior_type == PriorTypes::macauone)
            col_sideinfo = session->config.m_col_features;

         return ::create_prior(session, mode, session->config.col_prior_type, col_sideinfo);
      }
      default:
      {
         THROWERROR("Unknown prior mode");
      }
   }
}