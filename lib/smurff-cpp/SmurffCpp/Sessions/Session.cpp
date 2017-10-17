#include "Session.h"

#include <string>

#include <Eigen/Core>

#include <SmurffCpp/Priors/MacauOnePrior.hpp>
#include <SmurffCpp/Priors/MacauPrior.hpp>
#include <SmurffCpp/Priors/NormalPrior.h>
#include <SmurffCpp/Priors/SpikeAndSlabPrior.h>

#include <SmurffCpp/DataMatrices/MatrixDataFactory.h>

using namespace smurff;
using namespace Eigen;

template<class SideInfo>
inline void addMacauPrior(Session &m, std::string prior_name, SideInfo *f, double lambda_beta, double tol, int use_FtF)
{
   std::unique_ptr<SideInfo> features(f);

   if(prior_name == "macau" || prior_name == "default")
   {
      auto &prior = m.addPrior<MacauPrior<SideInfo>>();
      prior.addSideInfo(features, use_FtF);
      prior.setLambdaBeta(lambda_beta);
      prior.setTol(tol);
   } 
   else if(prior_name == "macauone") 
   {
      auto &prior = m.addPrior<MacauOnePrior<SideInfo>>();
      prior.addSideInfo(features, use_FtF);
      prior.setLambdaBeta(lambda_beta);
   } 
   else 
   {
      throw std::runtime_error("Unknown prior with side info: " + prior_name);
   }
}

void add_prior(Session &sess, int mode, std::string prior_name, const std::vector<MatrixConfig> &features, double lambda_beta, double tol, bool direct)
{
   //-- row prior with side information
   if (features.size())
   {
      if (prior_name == "macau" || prior_name == "macauone")
      {
         assert(features.size() == 1);
         auto &s = features.at(0);

         if (s.isBinary())
         {
            std::uint64_t nrow = s.getNRow();
            std::uint64_t ncol = s.getNCol();
            std::uint64_t nnz = s.getNNZ();
            std::shared_ptr<std::vector<std::uint32_t> > rows = s.getRowsPtr();
            std::shared_ptr<std::vector<std::uint32_t> > cols = s.getColsPtr();

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
            // macau expects the rows of the matrix to be equal to the mode
            // size, if the mode == 1 (col_features) we need to swap the rows and columns 
            if (mode == 1) {
                std::swap(nrow, ncol);
                std::swap(rowsRawPtr, colsRawPtr);
            }

            auto sideinfo = new SparseFeat(nrow, ncol, nnz, rowsRawPtr, colsRawPtr);
            addMacauPrior(sess, prior_name, sideinfo, lambda_beta, tol, direct);
         }
         else if (s.isDense())
         {
            auto sideinfo = dense_to_eigen(s);

            // Temporary solution #2
            // macau expects the rows of the matrix to be equal to the mode
            // size, if the mode == 1 (col_features) we need to swap the rows and columns 
            if (mode == 1) sideinfo.transposeInPlace();

            addMacauPrior(sess, prior_name, new MatrixXd(sideinfo), lambda_beta, tol, direct);
         }
         else
         {
            std::uint64_t nrow = s.getNRow();
            std::uint64_t ncol = s.getNCol();
            std::uint64_t nnz = s.getNNZ();
            std::shared_ptr<std::vector<std::uint32_t> > rows = s.getRowsPtr();
            std::shared_ptr<std::vector<std::uint32_t> > cols = s.getColsPtr();
            std::shared_ptr<std::vector<double> > values = s.getValuesPtr();

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
            if (mode == 1) {
                std::swap(nrow, ncol);
                std::swap(rowsRawPtr, colsRawPtr);
            }


            auto sideinfo = new SparseDoubleFeat(nrow, ncol, nnz, rowsRawPtr, colsRawPtr, valuesRawPtr);
            addMacauPrior(sess, prior_name, sideinfo, lambda_beta, tol, direct);
         }
      }
      else
      {
         assert(false && "SideInfo only with macau(one) prior");
      }
   }
   else if(prior_name == "normal" || prior_name == "default")
   {
      sess.addPrior<NormalPrior>();
   }
   else if(prior_name == "spikeandslab")
   {
      sess.addPrior<SpikeAndSlabPrior>();
   }
   else
   {
      throw std::runtime_error("Unknown prior without side info: " + prior_name);
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

   pred.set(sparse_to_eigen(config.test));

   std::vector<MatrixConfig> row_matrices;
   std::vector<MatrixConfig> col_matrices;

   std::vector<MatrixConfig> row_sideinfo;
   std::vector<MatrixConfig> col_sideinfo;

   if (config.row_prior == "macau" || config.row_prior == "macauone") 
      row_sideinfo = config.row_features;
   else 
      row_matrices = config.row_features;

   if (config.col_prior == "macau" || config.col_prior == "macauone") 
      col_sideinfo = config.col_features;
   else 
      col_matrices = config.col_features;

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


   // center mode
   data().setCenterMode(config.center_mode);


   add_prior(*this, 0, config.row_prior, row_sideinfo, config.lambda_beta, config.tol, config.direct);
   add_prior(*this, 1, config.col_prior, col_sideinfo, config.lambda_beta, config.tol, config.direct);
}

void Session::init() 
{
   threads_init();
   init_bmrng();
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
   os << indent << "  Version: " << Config::version() << "\n" ;
   os << indent << "  Iterations: " << config.burnin << " burnin + " << config.nsamples << " samples\n";
   
   if (config.save_freq > 0) 
   {
      os << indent << "  Save model: every " << config.save_freq << " iteration\n";
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
   if (!config.save_freq || isample < 0) 
      return;
   if (((isample+1) % config.save_freq) != 0) 
      return;

   std::string fprefix = config.save_prefix + "-sample-" + std::to_string(isample);

   if (config.verbose) 
      printf("-- Saving model, predictions,... into '%s*%s'.\n", fprefix.c_str(), config.save_suffix.c_str());

   BaseSession::save(fprefix, config.save_suffix);
}

void Session::printStatus(double elapsedi) 
{
   pred.update(model, data(), iter < config.burnin);

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
      double colmean_rmse = pred.rmse_using_modemean(data(), 0);
      double globalmean_rmse = pred.rmse_using_modemean(data(), 1);
      double train_rmse = data().train_rmse(model);

      auto f = fopen(config.csv_status.c_str(), "a");

      fprintf(f, "%s;%d;%d;%.4f;%.4f;%.4f;%.4f;%.4f;%.4f;:%.4f;%1.2e;%1.2e;%0.1f\n",
            phase.c_str(), i, from,
            globalmean_rmse, colmean_rmse,
            pred.rmse_avg, pred.rmse_1sample, train_rmse, pred.auc_1sample, pred.auc_avg, snorm0, snorm1, elapsedi);
            
      fclose(f);
   }
}
