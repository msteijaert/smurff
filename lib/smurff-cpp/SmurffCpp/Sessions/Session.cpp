#include "Session.h"

#include <string>

#include <Eigen/Core>

#include <SmurffCpp/Version.h>

#include <SmurffCpp/Utils/omp_util.h>

#include <SmurffCpp/Priors/MacauOnePrior.hpp>
#include <SmurffCpp/Priors/MacauPrior.hpp>
#include <SmurffCpp/Priors/NormalPrior.h>
#include <SmurffCpp/Priors/SpikeAndSlabPrior.h>

#include <SmurffCpp/DataMatrices/MatrixDataFactory.h>

using namespace smurff;
using namespace Eigen;

template<class SideInfo>
inline void addMacauPrior(Session &session, PriorTypes prior_type, std::shared_ptr<SideInfo> side_info, double lambda_beta, double tol, int use_FtF)
{
   if(prior_type == PriorTypes::macau || prior_type == PriorTypes::default_prior)
   {
      auto &prior = session.addPrior<MacauPrior<SideInfo>>();
      prior.addSideInfo(side_info, use_FtF);
      prior.setLambdaBeta(lambda_beta);
      prior.setTol(tol);
   }
   else if(prior_type == PriorTypes::macauone)
   {
      auto &prior = session.addPrior<MacauOnePrior<SideInfo>>();
      prior.addSideInfo(side_info, use_FtF);
      prior.setLambdaBeta(lambda_beta);
   }
   else
   {
      throw std::runtime_error("Unknown prior with side info: " + priorTypeToString(prior_type));
   }
}

std::shared_ptr<SparseFeat> side_info_config_to_sparse_binary_features(const MatrixConfig& sideinfoConfig, int mode)
{
   std::uint64_t nrow = sideinfoConfig.getNRow();
   std::uint64_t ncol = sideinfoConfig.getNCol();
   std::uint64_t nnz = sideinfoConfig.getNNZ();

   std::shared_ptr<std::vector<std::uint32_t> > rows = sideinfoConfig.getRowsPtr();
   std::shared_ptr<std::vector<std::uint32_t> > cols = sideinfoConfig.getColsPtr();

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

std::shared_ptr<Eigen::MatrixXd> side_info_config_to_dense_features(const MatrixConfig& sideinfoConfig, int mode)
{
   Eigen::MatrixXd sideinfo = matrix_utils::dense_to_eigen(sideinfoConfig);
   
   // Temporary solution #2
   // macau expects the rows of the matrix to be equal to the mode size, 
   // if the mode == 1 (col_features) we need to swap the rows and columns
   if (mode == 1) 
      sideinfo.transposeInPlace();

   return std::shared_ptr<Eigen::MatrixXd>(new Eigen::MatrixXd(sideinfo));
}

std::shared_ptr<SparseDoubleFeat> side_info_config_to_sparse_features(const MatrixConfig& sideinfoConfig, int mode)
{
   std::uint64_t nrow = sideinfoConfig.getNRow();
   std::uint64_t ncol = sideinfoConfig.getNCol();
   std::uint64_t nnz = sideinfoConfig.getNNZ();

   std::shared_ptr<std::vector<std::uint32_t> > rows = sideinfoConfig.getRowsPtr();
   std::shared_ptr<std::vector<std::uint32_t> > cols = sideinfoConfig.getColsPtr();
   std::shared_ptr<std::vector<double> > values = sideinfoConfig.getValuesPtr();

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

void add_macau_prior(Session &session, int mode, PriorTypes prior_type, const std::vector<MatrixConfig>& vsideinfo, double lambda_beta, double tol, bool direct)
{
   if(vsideinfo.size() != 1)
      throw std::runtime_error("Only one feature matrix is allowed");

   const MatrixConfig& sideinfoConfig = vsideinfo.at(0);

   if (sideinfoConfig.isBinary())
   {
      std::shared_ptr<SparseFeat> sideinfo = side_info_config_to_sparse_binary_features(sideinfoConfig, mode);
      addMacauPrior(session, prior_type, sideinfo, lambda_beta, tol, direct);
   }
   else if (sideinfoConfig.isDense())
   {
      std::shared_ptr<Eigen::MatrixXd> sideinfo = side_info_config_to_dense_features(sideinfoConfig, mode);
      addMacauPrior(session, prior_type, sideinfo, lambda_beta, tol, direct);
   }
   else
   {
      std::shared_ptr<SparseDoubleFeat> sideinfo = side_info_config_to_sparse_features(sideinfoConfig, mode);
      addMacauPrior(session, prior_type, sideinfo, lambda_beta, tol, direct);
   }
}

void add_prior(Session& session, int mode, PriorTypes prior_type, const std::vector<MatrixConfig>& vsideinfo, double lambda_beta, double tol, bool direct)
{
   // row prior with side information
   // side information can only be applied to macau and macauone priors
   if (vsideinfo.size())
   {
      switch(prior_type)
      {
      case PriorTypes::macau:
      case PriorTypes::macauone:
         add_macau_prior(session, mode, prior_type, vsideinfo, lambda_beta, tol, direct);
         break;
      default:
         throw std::runtime_error("SideInfo only with macau(one) prior");
      }
   }
   else
   {
      switch(prior_type)
      {
      case PriorTypes::normal:
      case PriorTypes::default_prior:
         session.addPrior<NormalPrior>();
         break;
      case PriorTypes::spikeandslab:
         session.addPrior<SpikeAndSlabPrior>();
         break;
      default:
         throw std::runtime_error("Unknown prior without side info: " + priorTypeToString(prior_type));
      }
   }
}

void Session::setFromConfig(const Config &c)
{
   c.validate(true);
   c.save(config.save_prefix + ".ini");

   //-- copy
   config = c;

   if (config.classify)
      pred.setThreshold(config.threshold);

   pred.set(matrix_utils::sparse_to_eigen(config.test));

   std::vector<MatrixConfig> row_matrices;
   std::vector<MatrixConfig> col_matrices;

   std::vector<MatrixConfig> row_sideinfo;
   std::vector<MatrixConfig> col_sideinfo;

   if (config.row_prior_type == PriorTypes::macau || config.row_prior_type == PriorTypes::macauone)
      row_sideinfo = config.row_features;
   else
      row_matrices = config.row_features;

   if (config.col_prior_type == PriorTypes::macau || config.col_prior_type == PriorTypes::macauone)
      col_sideinfo = config.col_features;
   else
      col_matrices = config.col_features;

   //row_matrices and col_matrices are selected if prior is not macau and not macauone
   data_ptr = smurff::matrix_config_to_matrix(config.train, row_matrices, col_matrices);

   // check if data is ScarceBinary
   /*
   if (0)
      if (!config.classify) {
            config.classify = true;
            config.threshold = 0.5;
            config.train.noise.name = "probit";
      }
   */

   //row_sideinfo and col_sideinfo are selected if prior is macau or macauone
   add_prior(*this, 0, config.row_prior_type, row_sideinfo, config.lambda_beta, config.tol, config.direct);
   add_prior(*this, 1, config.col_prior_type, col_sideinfo, config.lambda_beta, config.tol, config.direct);
}

void Session::init()
{
    threads_init();
    if (config.random_seed_set) init_bmrng(config.random_seed);
    else init_bmrng();
    
    data().init();
    model.init(config.num_latent, data().dim(), config.init_model);
    for( auto &p : priors)
        p->init();
    
    if (config.csv_status.size())
    {
        auto f = fopen(config.csv_status.c_str(), "w");
        fprintf(f, "phase;iter;phase_len;globmean_rmse;colmean_rmse;rmse_avg;rmse_1samp;train_rmse;auc_avg;auc_1samp;U0;U1;elapsed\n");
        fclose(f);
    }
    
    if (config.verbose)
        info(std::cout, "");
    
    if (config.restore_prefix.size())
    {
        if (config.verbose)
            printf("-- Restoring model, predictions,... from '%s*%s'.\n", config.restore_prefix.c_str(), config.save_suffix.c_str());
        restore(config.restore_prefix, config.restore_suffix);
    }
    
    if (config.verbose)
    {
        printStatus(0);
        printf(" ====== Sampling (burning phase) ====== \n");
    }
    
    iter = 0;
    is_init = true;
}

void Session::run()
{
   init();
   while (iter < config.burnin + config.nsamples) step();
}

void Session::step()
{
   assert(is_init);

   if (config.verbose && iter == config.burnin)
   {
      printf(" ====== Burn-in complete, averaging samples ====== \n");
   }

   auto starti = tick();
   BaseSession::step();
   auto endi = tick();

   printStatus(endi - starti);
   save(iter - config.burnin + 1);
   iter++;
}

std::ostream& Session::info(std::ostream &os, std::string indent)
{
   BaseSession::info(os, indent);
   os << indent << "  Version: " << smurff::SMURFF_VERSION << "\n" ;
   os << indent << "  Iterations: " << config.burnin << " burnin + " << config.nsamples << " samples\n";

   if (config.save_freq != 0)
   {
      if (config.save_freq > 0) {
          os << indent << "  Save model: every " << config.save_freq << " iteration\n";
      } else {
          os << indent << "  Save model after last iteration\n";
      }
      os << indent << "  Save prefix: " << config.save_prefix << "\n";
      os << indent << "  Save suffix: " << config.save_suffix << "\n";
   }
   else
   {
      os << indent << "  Save model: never\n";
   }

   if (config.restore_prefix.size())
   {
      os << indent << "  Restore prefix: " << config.restore_prefix << "\n";
      os << indent << "  Restore suffix: " << config.restore_suffix << "\n";
   }

   os << indent << "}\n";
   return os;
}

void Session::save(int isample)
{
   if (!config.save_freq || isample < 0) return;
   
   //save_freq > 0: check modulo
   if (config.save_freq > 0 && ((isample+1) % config.save_freq) != 0) return;
   //save_freq < 0: save last iter
   if (config.save_freq < 0 && isample < config.nsamples) return;

   std::string fprefix = config.save_prefix + "-sample-" + std::to_string(isample);

   if (config.verbose)
      printf("-- Saving model, predictions,... into '%s*%s'.\n", fprefix.c_str(), config.save_suffix.c_str());

   BaseSession::save(fprefix, config.save_suffix);
}

void Session::printStatus(double elapsedi)
{
   pred.update(model, iter < config.burnin);

   if(!config.verbose)
      return;

   double snorm0 = model.U(0).norm();
   double snorm1 = model.U(1).norm();

   auto nnz_per_sec = (data().nnz()) / elapsedi;
   auto samples_per_sec = (model.nsamples()) / elapsedi;

   std::string phase;
   int i, from;
   if (iter < 0)
   {
      phase = "Initial";
      i = 0;
      from = 0;
   }
   else if (iter < config.burnin)
   {
      phase = "Burnin";
      i = iter + 1;
      from = config.burnin;
   }
   else
   {
      phase = "Sample";
      i = iter - config.burnin + 1;
      from = config.nsamples;
   }

   printf("%s %3d/%3d: RMSE: %.4f (1samp: %.4f)", phase.c_str(), i, from, pred.rmse_avg, pred.rmse_1sample);

   if (config.classify)
      printf(" AUC:%.4f (1samp: %.4f)", pred.auc_avg, pred.auc_1sample);

   printf("  U:[%1.2e, %1.2e] [took: %0.1fs]\n", snorm0, snorm1, elapsedi);

   if (config.verbose > 1)
   {
      double train_rmse = data().train_rmse(model);
      printf("  RMSE train: %.4f\n", train_rmse);
      printf("  Priors:\n");

      for(const auto &p : priors)
         p->status(std::cout, "     ");

      printf("  Model:\n");
      model.status(std::cout, "    ");
      printf("  Noise:\n");
      data().status(std::cout, "    ");
   }

   if (config.verbose > 2)
   {
      printf("  Compute Performance: %.0f samples/sec, %.0f nnz/sec\n", samples_per_sec, nnz_per_sec);
   }

   if (config.csv_status.size())
   {
      double train_rmse = data().train_rmse(model);

      auto f = fopen(config.csv_status.c_str(), "a");

      fprintf(f, "%s;%d;%d;%.4f;%.4f;%.4f;%.4f;:%.4f;%1.2e;%1.2e;%0.1f\n",
            phase.c_str(), i, from,            
            pred.rmse_avg, pred.rmse_1sample, train_rmse, pred.auc_1sample, pred.auc_avg, snorm0, snorm1, elapsedi);

      fclose(f);
   }
}
