#include "Session.h"

#include <string>

#include <SmurffCpp/Version.h>

#include <SmurffCpp/Utils/omp_util.h>
#include <SmurffCpp/Utils/Distribution.h>
#include <SmurffCpp/Utils/MatrixUtils.h>
#include <SmurffCpp/Utils/counters.h>
#include <SmurffCpp/Utils/Error.h>

#include <SmurffCpp/DataMatrices/DataCreator.h>
#include <SmurffCpp/Priors/PriorFactory.h>

#include <SmurffCpp/result.h>

using namespace smurff;

void Session::setFromConfig(const Config& cfg)
{
   // assign config

   cfg.validate();
   cfg.save(cfg.getSavePrefix() + ".ini");
   config = cfg;

   std::shared_ptr<Session> this_session = shared_from_this();

   // initialize pred

   if (config.getClassify())
      m_pred->setThreshold(config.getThreshold());

   if (config.getTest())
      m_pred->set(config.getTest());

   // initialize data

   data_ptr = config.getTrain()->create(std::make_shared<DataCreator>(this_session));

   // initialize priors

   std::shared_ptr<IPriorFactory> priorFactory = this->create_prior_factory();
   for(std::size_t i = 0; i < config.getPriorTypes().size(); i++)
      this->addPrior(priorFactory->create_prior(this_session, i));
}

void Session::init()
{
   //init omp
   threads_init();

   //init random generator
   if(config.getRandomSeedSet())
      init_bmrng(config.getRandomSeed());
   else
      init_bmrng();

   //initialize train matrix (centring and noise model)
   data()->init();

   //initialize model (samples)
   m_model->init(config.getNumLatent(), data()->dim(), config.getModelInitType());

   //initialize priors (?)
   for( auto &p : m_priors)
      p->init();

   //write header to status file
   if (config.getCsvStatus().size())
   {
      auto f = fopen(config.getCsvStatus().c_str(), "w");
      fprintf(f, "phase;iter;phase_len;globmean_rmse;colmean_rmse;rmse_avg;rmse_1samp;train_rmse;auc_avg;auc_1samp;U0;U1;elapsed\n");
      fclose(f);
   }

   //write info to console
   if (config.getVerbose())
      info(std::cout, "");

   //restore session (model, priors)
   if (config.getRestorePrefix().size())
   {
      if (config.getVerbose())
         printf("-- Restoring model, predictions,... from '%s*%s'.\n", config.getRestorePrefix().c_str(), config.getSaveSuffix().c_str());
      restore(config.getRestorePrefix(), config.getRestoreSuffix());
   }

   //print session status to console
   if (config.getVerbose())
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
   while (iter < config.getBurnin() + config.getNSamples())
      step();
}

void Session::step()
{
   THROWERROR_ASSERT(is_init);

   if (config.getVerbose() && iter == config.getBurnin())
   {
      printf(" ====== Burn-in complete, averaging samples ====== \n");
   }

   auto starti = tick();
   BaseSession::step();
   auto endi = tick();

   //WARNING: update is an expensive operation because of sort (when calculating AUC)
   m_pred->update(m_model, iter < config.getBurnin());

   printStatus(endi - starti);
   save(iter - config.getBurnin() + 1);
   iter++;
}

std::ostream& Session::info(std::ostream &os, std::string indent)
{
   BaseSession::info(os, indent);
   os << indent << "  Version: " << smurff::SMURFF_VERSION << "\n" ;
   os << indent << "  Iterations: " << config.getBurnin() << " burnin + " << config.getNSamples() << " samples\n";

   if (config.getSaveFreq() != 0)
   {
      if (config.getSaveFreq() > 0) {
          os << indent << "  Save model: every " << config.getSaveFreq() << " iteration\n";
      } else {
          os << indent << "  Save model after last iteration\n";
      }
      os << indent << "  Save prefix: " << config.getSavePrefix() << "\n";
      os << indent << "  Save suffix: " << config.getSaveSuffix() << "\n";
   }
   else
   {
      os << indent << "  Save model: never\n";
   }

   if (config.getRestorePrefix().size())
   {
      os << indent << "  Restore prefix: " << config.getRestorePrefix() << "\n";
      os << indent << "  Restore suffix: " << config.getRestoreSuffix() << "\n";
   }

   os << indent << "}\n";
   return os;
}

void Session::save(int isample)
{
   if (!config.getSaveFreq() || isample < 0) //do not save if (never save) mode is selected or if burnin
      return;

   //save_freq > 0: check modulo
   if (config.getSaveFreq() > 0 && ((isample + 1) % config.getSaveFreq()) != 0) //do not save if not a save iteration
      return;

   //save_freq < 0: save last iter
   if (config.getSaveFreq() < 0 && isample < config.getNSamples()) //do not save if (final model) mode is selected and not a final iteration
      return;

   std::string fprefix = config.getSavePrefix() + "-sample-" + std::to_string(isample);

   if (config.getVerbose())
      printf("-- Saving model, predictions,... into '%s*%s'.\n", fprefix.c_str(), config.getSaveSuffix().c_str());

   BaseSession::save(fprefix, config.getSaveSuffix());
}

void Session::printStatus(double elapsedi)
{
   if(!config.getVerbose())
      return;

   double snorm0 = m_model->U(0).norm();
   double snorm1 = m_model->U(1).norm();

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
   else if (iter < config.getBurnin())
   {
      phase = "Burnin";
      i = iter + 1;
      from = config.getBurnin();
   }
   else
   {
      phase = "Sample";
      i = iter - config.getBurnin() + 1;
      from = config.getNSamples();
   }

   printf("%s %3d/%3d: RMSE: %.4f (1samp: %.4f)", phase.c_str(), i, from, m_pred->rmse_avg, m_pred->rmse_1sample);

   if (config.getClassify())
      printf(" AUC:%.4f (1samp: %.4f)", m_pred->auc_avg, m_pred->auc_1sample);

   printf("  U:[%1.2e, %1.2e] [took: %0.1fs]\n", snorm0, snorm1, elapsedi);

   // avoid computing train_rmse twice
   double train_rmse = NAN;

   if (config.getVerbose() > 1)
   {
      train_rmse = data()->train_rmse(m_model);
      printf("  RMSE train: %.4f\n", train_rmse);
      printf("  Priors:\n");

      for(const auto &p : m_priors)
         p->status(std::cout, "     ");

      printf("  Model:\n");
      m_model->status(std::cout, "    ");
      printf("  Noise:\n");
      data()->status(std::cout, "    ");
   }

   if (config.getVerbose() > 2)
   {
      printf("  Compute Performance: %.0f samples/sec, %.0f nnz/sec\n", samples_per_sec, nnz_per_sec);
   }

   if (config.getCsvStatus().size())
   {
      // train_rmse is printed as NAN, unless verbose > 1
      auto f = fopen(config.getCsvStatus().c_str(), "a");
      fprintf(f, "%s;%d;%d;%.4f;%.4f;%.4f;%.4f;:%.4f;%1.2e;%1.2e;%0.1f\n",
                  phase.c_str(), i, from,
                  m_pred->rmse_avg, m_pred->rmse_1sample, train_rmse, m_pred->auc_1sample, m_pred->auc_avg, snorm0, snorm1, elapsedi);
      fclose(f);
   }
}

std::shared_ptr<IPriorFactory> Session::create_prior_factory() const
{
   return std::make_shared<PriorFactory>();
}
