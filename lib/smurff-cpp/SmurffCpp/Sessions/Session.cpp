#include "Session.h"

#include <string>

#include <SmurffCpp/Version.h>

#include <SmurffCpp/Utils/omp_util.h>
#include <SmurffCpp/Utils/Distribution.h>
#include <SmurffCpp/Utils/MatrixUtils.h>

#include <SmurffCpp/DataMatrices/DataCreator.h>
#include <SmurffCpp/Priors/PriorFactory.h>

#include <SmurffCpp/result.h>

using namespace smurff;

void Session::setFromConfig(const Config& cfg)
{
   // assign config

   cfg.validate(true);
   cfg.save(cfg.getSavePrefix() + ".ini");
   config = cfg;

   std::shared_ptr<Session> this_session = shared_from_this();

   // initialize pred

   if (config.classify)
      m_pred->setThreshold(config.threshold);

   m_pred->set(config.m_test);

   // initialize data

   data_ptr = config.m_train->create(std::make_shared<DataCreator>(this_session));

   // check if data is ScarceBinary
   /*
   if (0)
      if (!config.classify) {
            config.classify = true;
            config.threshold = 0.5;
            config.train.noise.name = "probit";
      }
   */

   // initialize priors

   this->addPrior(PriorFactory::create_prior(this_session, 0));

   this->addPrior(PriorFactory::create_prior(this_session, 1));
}

void Session::init()
{
   //init omp
   threads_init();

   //init random generator
   if(config.random_seed_set)
      init_bmrng(config.random_seed);
   else
      init_bmrng();

   //initialize train matrix (centring and noise model)
   data()->init();

   //initialize model (samples)
   m_model->init(config.num_latent, data()->dim(), config.model_init_type);

   //initialize priors (?)
   for( auto &p : m_priors)
      p->init();

   //write header to status file
   if (config.csv_status.size())
   {
      auto f = fopen(config.csv_status.c_str(), "w");
      fprintf(f, "phase;iter;phase_len;globmean_rmse;colmean_rmse;rmse_avg;rmse_1samp;train_rmse;auc_avg;auc_1samp;U0;U1;elapsed\n");
      fclose(f);
   }

   //write info to console
   if (config.verbose)
      info(std::cout, "");

   //restore session (model, priors)
   if (config.restore_prefix.size())
   {
      if (config.verbose)
         printf("-- Restoring model, predictions,... from '%s*%s'.\n", config.restore_prefix.c_str(), config.save_suffix.c_str());
      restore(config.restore_prefix, config.restore_suffix);
   }

   //print session status to console
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
   while (iter < config.burnin + config.nsamples)
      step();
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

   //WARNING: update is an expensive operation because of sort (when calculating AUC)
   m_pred->update(m_model, iter < config.burnin);

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
      os << indent << "  Save prefix: " << config.getSavePrefix() << "\n";
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
   if (!config.save_freq || isample < 0) //do not save if (never save) mode is selected or if burnin
      return;

   //save_freq > 0: check modulo
   if (config.save_freq > 0 && ((isample + 1) % config.save_freq) != 0) //do not save if not a save iteration
      return;

   //save_freq < 0: save last iter
   if (config.save_freq < 0 && isample < config.nsamples) //do not save if (final model) mode is selected and not a final iteration
      return;

   std::string fprefix = config.getSavePrefix() + "-sample-" + std::to_string(isample);

   if (config.verbose)
      printf("-- Saving model, predictions,... into '%s*%s'.\n", fprefix.c_str(), config.save_suffix.c_str());

   BaseSession::save(fprefix, config.save_suffix);
}

void Session::printStatus(double elapsedi)
{
   if(!config.verbose)
      return;

   double snorm0 = m_model->U(0)->norm();
   double snorm1 = m_model->U(1)->norm();

   auto nnz_per_sec = (data()->nnz()) / elapsedi;
   auto samples_per_sec = (m_model->nsamples()) / elapsedi;

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

   printf("%s %3d/%3d: RMSE: %.4f (1samp: %.4f)", phase.c_str(), i, from, m_pred->rmse_avg, m_pred->rmse_1sample);

   if (config.classify)
      printf(" AUC:%.4f (1samp: %.4f)", m_pred->auc_avg, m_pred->auc_1sample);

   printf("  U:[%1.2e, %1.2e] [took: %0.1fs]\n", snorm0, snorm1, elapsedi);

   if (config.verbose > 1)
   {
      double train_rmse = data()->train_rmse(m_model);
      printf("  RMSE train: %.4f\n", train_rmse);
      printf("  Priors:\n");

      for(const auto &p : m_priors)
         p->status(std::cout, "     ");

      printf("  Model:\n");
      m_model->status(std::cout, "    ");
      printf("  Noise:\n");
      data()->status(std::cout, "    ");
   }

   if (config.verbose > 2)
   {
      printf("  Compute Performance: %.0f samples/sec, %.0f nnz/sec\n", samples_per_sec, nnz_per_sec);
   }

   if (config.csv_status.size())
   {
      double train_rmse = data()->train_rmse(m_model);
      auto f = fopen(config.csv_status.c_str(), "a");
      fprintf(f, "%s;%d;%d;%.4f;%.4f;%.4f;%.4f;:%.4f;%1.2e;%1.2e;%0.1f\n",
                  phase.c_str(), i, from,
                  m_pred->rmse_avg, m_pred->rmse_1sample, train_rmse, m_pred->auc_1sample, m_pred->auc_avg, snorm0, snorm1, elapsedi);
      fclose(f);
   }
}
