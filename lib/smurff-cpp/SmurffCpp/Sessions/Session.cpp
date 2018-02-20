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

void Session::setFromRootPath(std::string rootPath)
{
   // assign config

   m_rootFile = std::make_shared<RootFile>(rootPath);
   m_rootFile->restoreConfig(m_config);

   m_config.validate();

   //base functionality
   setFromBase();
}

void Session::setFromConfig(const Config& cfg)
{
   // assign config

   cfg.validate();
   m_config = cfg;

   m_rootFile = std::make_shared<RootFile>(m_config.getSavePrefix(), m_config.getSaveExtension());
   m_rootFile->saveConfig(m_config);

   //base functionality
   setFromBase();
}

void Session::setFromBase()
{
   std::shared_ptr<Session> this_session = shared_from_this();

   // initialize pred

   if (m_config.getClassify())
      m_pred->setThreshold(m_config.getThreshold());

   if (m_config.getTest())
      m_pred->set(m_config.getTest());

   // initialize data

   data_ptr = m_config.getTrain()->create(std::make_shared<DataCreator>(this_session));

   // initialize priors

   std::shared_ptr<IPriorFactory> priorFactory = this->create_prior_factory();
   for (std::size_t i = 0; i < m_config.getPriorTypes().size(); i++)
      this->addPrior(priorFactory->create_prior(this_session, i));
}

void Session::init()
{
   m_iter = 0;

   //init omp
   threads_init();

   //initialize random generator
   initRng();

   //initialize train matrix (centring and noise model)
   data()->init();

   //initialize model (samples)
   m_model->init(m_config.getNumLatent(), data()->dim(), m_config.getModelInitType());

   //initialize priors
   for(auto &p : m_priors)
      p->init();

   //write header to status file
   if (m_config.getCsvStatus().size())
   {
      auto f = fopen(m_config.getCsvStatus().c_str(), "w");
      fprintf(f, "phase;iter;phase_len;globmean_rmse;colmean_rmse;rmse_avg;rmse_1samp;train_rmse;auc_avg;auc_1samp;U0;U1;elapsed\n");
      fclose(f);
   }

   //write info to console
   if (m_config.getVerbose())
      info(std::cout, "");

   //restore session (model, priors)
   bool resume = restore();

   //print session status to console
   if (m_config.getVerbose())
   {
      printStatus(0, resume);
      printf(" ====== Sampling (burning phase) ====== \n");
   }

   is_init = true;
}

void Session::run()
{
   init();

   while (m_iter < m_config.getBurnin() + m_config.getNSamples())
      step();
}

void Session::step()
{
   THROWERROR_ASSERT(is_init);

   if (m_config.getVerbose() && m_iter == m_config.getBurnin())
   {
      printf(" ====== Burn-in complete, averaging samples ====== \n");
   }

   auto starti = tick();
   BaseSession::step();
   auto endi = tick();

   //WARNING: update is an expensive operation because of sort (when calculating AUC)
   m_pred->update(m_model, m_iter < m_config.getBurnin());

   printStatus(endi - starti, false);

   save(m_iter - m_config.getBurnin() + 1);

   m_iter++;
}

std::ostream& Session::info(std::ostream &os, std::string indent)
{
   os << indent << name << " {\n";

   BaseSession::info(os, indent);

   os << indent << "  Version: " << smurff::SMURFF_VERSION << "\n" ;
   os << indent << "  Iterations: " << m_config.getBurnin() << " burnin + " << m_config.getNSamples() << " samples\n";

   if (m_config.getSaveFreq() != 0)
   {
      if (m_config.getSaveFreq() > 0) 
      {
          os << indent << "  Save model: every " << m_config.getSaveFreq() << " iteration\n";
      } 
      else 
      {
          os << indent << "  Save model after last iteration\n";
      }

      os << indent << "  Save prefix: " << m_config.getSavePrefix() << "\n";
      os << indent << "  Save extension: " << m_config.getSaveExtension() << "\n";
   }
   else
   {
      os << indent << "  Save model: never\n";
   }

   os << indent << "}\n";
   return os;
}

void Session::save(int isample) const
{
   if (!m_config.getSaveFreq() || isample < 0) //do not save if (never save) mode is selected or if burnin
      return;

   //save_freq > 0: check modulo
   if (m_config.getSaveFreq() > 0 && ((isample + 1) % m_config.getSaveFreq()) != 0) //do not save if not a save iteration
      return;

   //save_freq < 0: save last iter
   if (m_config.getSaveFreq() < 0 && isample < m_config.getNSamples()) //do not save if (final model) mode is selected and not a final iteration
      return;

   std::shared_ptr<StepFile> stepFile = m_rootFile->createStepFile(isample);

   if (m_config.getVerbose())
      printf("-- Saving model, predictions,... into '%s'.\n", stepFile->getStepFileName().c_str());

   BaseSession::save(stepFile);

   //flush last item in a root file
   m_rootFile->flushLast();
}

bool Session::restore()
{
   std::shared_ptr<StepFile> stepFile = m_rootFile->openLastStepFile();
   if (!stepFile)
      return false;

   if (m_config.getVerbose())
      printf("-- Restoring model, predictions,... from '%s'.\n", stepFile->getStepFileName().c_str());

   BaseSession::restore(stepFile);

   //restore last iteration index
   m_iter = stepFile->getIsample() + m_config.getBurnin() - 1; //restore original state
   m_iter++; //go to next iteration

   return true;
}

void Session::printStatus(double elapsedi, bool resume)
{
   if(!m_config.getVerbose())
      return;

   double snorm0 = m_model->U(0).norm();
   double snorm1 = m_model->U(1).norm();

   auto nnz_per_sec = (data()->nnz()) / elapsedi;
   auto samples_per_sec = (m_model->nsamples()) / elapsedi;

   std::string resumeString = resume ? "Continue from " : std::string();

   std::string phase;
   int i, from;
   if (m_iter < 0)
   {
      phase = "Initial";
      i = 0;
      from = 0;
   }
   else if (m_iter < m_config.getBurnin())
   {
      phase = "Burnin";
      i = m_iter + 1;
      from = m_config.getBurnin();
   }
   else
   {
      phase = "Sample";
      i = m_iter - m_config.getBurnin() + 1;
      from = m_config.getNSamples();
   }

   printf("%s%s %3d/%3d: RMSE: %.4f (1samp: %.4f)", resumeString.c_str(), phase.c_str(), i, from, m_pred->rmse_avg, m_pred->rmse_1sample);

   if (m_config.getClassify())
      printf(" AUC:%.4f (1samp: %.4f)", m_pred->auc_avg, m_pred->auc_1sample);

   printf("  U:[%1.2e, %1.2e] [took: %0.1fs]\n", snorm0, snorm1, elapsedi);

   // avoid computing train_rmse twice
   double train_rmse = NAN;

   if (m_config.getVerbose() > 1)
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

   if (m_config.getVerbose() > 2)
   {
      printf("  Compute Performance: %.0f samples/sec, %.0f nnz/sec\n", samples_per_sec, nnz_per_sec);
   }

   if (m_config.getCsvStatus().size())
   {
      // train_rmse is printed as NAN, unless verbose > 1
      auto f = fopen(m_config.getCsvStatus().c_str(), "a");
      fprintf(f, "%s;%d;%d;%.4f;%.4f;%.4f;%.4f;:%.4f;%1.2e;%1.2e;%0.1f\n",
                  phase.c_str(), i, from,
                  m_pred->rmse_avg, m_pred->rmse_1sample, train_rmse, m_pred->auc_1sample, m_pred->auc_avg, snorm0, snorm1, elapsedi);
      fclose(f);
   }
}

void Session::initRng()
{
   //init random generator
   if (m_config.getRandomSeedSet())
      init_bmrng(m_config.getRandomSeed());
   else
      init_bmrng();
}

std::shared_ptr<IPriorFactory> Session::create_prior_factory() const
{
   return std::make_shared<PriorFactory>();
}