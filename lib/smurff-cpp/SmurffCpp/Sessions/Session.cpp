#include "Session.h"

#include <string>
#include <iomanip>

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

   //flush record about options.ini
   m_rootFile->flushLast();
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
   //init omp
   threads_init(m_config.getVerbose());

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
      THROWERROR_ASSERT_MSG(f, "Could not open status csv file: " + m_config.getCsvStatus());
      fprintf(f, "phase;iter;phase_len;rmse_avg;rmse_1samp;train_rmse;auc_avg;auc_1samp;U0;U1;elapsed\n");
      fclose(f);
   }

   //write info to console
   if (m_config.getVerbose())
      info(std::cout, "");

   //restore session (model, priors)
   bool resume = restore(m_iter);

   //print session status to console
   if (m_config.getVerbose())
   {
      printStatus(std::cout, 0, resume, m_iter);
   }

   //restore will either start from initial iteration (-1) that should be printed with printStatus
   //or it will start with last iteration that was previously saved
   //in any case - we have to move to next iteration
   m_iter++; //go to next iteration

   is_init = true;
}

void Session::run()
{
   init();
   while (step());
}

bool Session::step()
{
   bool isStep = m_iter < m_config.getBurnin() + m_config.getNSamples();

   if (isStep)
   {
      //init omp
      threads_enable(m_config.getVerbose());

      THROWERROR_ASSERT(is_init);

      auto starti = tick();
      BaseSession::step();
      auto endi = tick();

      //WARNING: update is an expensive operation because of sort (when calculating AUC)
      m_pred->update(m_model, m_iter < m_config.getBurnin());

      printStatus(std::cout, endi - starti, false, m_iter);

      save(m_iter);

      m_iter++;

      threads_disable(m_config.getVerbose());
   }

   return isStep;
}

std::ostream& Session::info(std::ostream &os, std::string indent)
{
   os << indent << name << " {\n";

   BaseSession::info(os, indent);

   os << indent << "  Version: " << smurff::SMURFF_VERSION << "\n" ;
   os << indent << "  Iterations: " << m_config.getBurnin() << " burnin + " << m_config.getNSamples() << " samples\n";

   if (m_config.getSaveFreq() != 0 || m_config.getCheckpointFreq() != 0)
   {
      if (m_config.getSaveFreq() > 0)
      {
          os << indent << "  Save model: every " << m_config.getSaveFreq() << " iteration\n";
      }
      else if (m_config.getSaveFreq() < 0)
      {
          os << indent << "  Save model after last iteration\n";
      }

      if (m_config.getCheckpointFreq() > 0)
      {
          os << indent << "  Checkpoint state: every " << m_config.getCheckpointFreq() << " seconds\n";
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

void Session::save(int iteration)
{
   //do not save if 'never save' mode is selected
   if (!m_config.getSaveFreq() && 
       !m_config.getCheckpointFreq() &&
       !m_config.getCsvStatus().size())
      return;

   std::int32_t isample = iteration - m_config.getBurnin() + 1;

   //save if checkpoint threshold overdue
   if (m_config.getCheckpointFreq() && (tick() - m_lastCheckpointTime) >= m_config.getCheckpointFreq())
   {
      std::int32_t icheckpoint = iteration + 1;

      //save this iteration
      std::shared_ptr<StepFile> stepFile = m_rootFile->createCheckpointStepFile(icheckpoint);
      saveInternal(stepFile);

      //remove previous iteration if required (initial m_lastCheckpointIter is -1 which means that it does not exist)
      if (m_lastCheckpointIter >= 0)
      {
         std::int32_t icheckpointPrev = m_lastCheckpointIter + 1;

         //remove previous iteration
         m_rootFile->removeCheckpointStepFile(icheckpointPrev);

         //flush last item in a root file
         m_rootFile->flushLast();
      }

      //upddate counters
      m_lastCheckpointTime = tick();
      m_lastCheckpointIter = iteration;
   } 

   //save model during sampling stage
   if (m_config.getSaveFreq() && isample > 0)
   {
      //save_freq > 0: check modulo - do not save if not a save iteration
      if (m_config.getSaveFreq() > 0 && (isample % m_config.getSaveFreq()) != 0)
      {
          // don't save
      }
      //save_freq < 0: save last iter - do not save if (final model) mode is selected and not a final iteration
      else if (m_config.getSaveFreq() < 0 && isample < m_config.getNSamples())
      {
          // don't save
      }
      else
      {
          //do save this iteration
          std::shared_ptr<StepFile> stepFile = m_rootFile->createSampleStepFile(isample);
          saveInternal(stepFile);
      }
   }
}

void Session::saveInternal(std::shared_ptr<StepFile> stepFile)
{
   if (m_config.getVerbose())
   {
      std::cout << "-- Saving model, predictions,... into '" << stepFile->getStepFileName() << "'." << std::endl;
   }

   BaseSession::save(stepFile);

   //flush last item in a root file
   m_rootFile->flushLast();
}

bool Session::restore(int& iteration)
{
   std::shared_ptr<StepFile> stepFile = m_rootFile->openLastStepFile();
   if (!stepFile)
   {
      //if there is nothing to restore - start from initial iteration
      iteration = -1;

      //to keep track at what time we last checkpointed
      m_lastCheckpointTime = tick();
      m_lastCheckpointIter = -1;
      return false;
   }
   else
   {
      if (m_config.getVerbose())
      {
         std::cout << "-- Restoring model, predictions,... from '" << stepFile->getStepFileName() << "'." << std::endl;
      }

      BaseSession::restore(stepFile);

      //restore last iteration index
      if (stepFile->getCheckpoint())
      {
         iteration = stepFile->getIsample() - 1; //restore original state

         //to keep track at what time we last checkpointed
         m_lastCheckpointTime = tick();
         m_lastCheckpointIter = iteration;
      }
      else
      {
         iteration = stepFile->getIsample() + m_config.getBurnin() - 1; //restore original state

         //to keep track at what time we last checkpointed
         m_lastCheckpointTime = tick();
         m_lastCheckpointIter = iteration;
      }

      return true;
   }
}

void Session::printStatus(std::ostream& output, double elapsedi, bool resume, int iteration)
{
   if(!m_config.getVerbose() &&
      !m_config.getCsvStatus().size())
      return;

   double snorm0 = m_model->U(0).norm();
   double snorm1 = m_model->U(1).norm();

   // avoid computing train_rmse twice
   double train_rmse = NAN;

   double nnz_per_sec =  (double)(data()->nnz()) / elapsedi;
   double samples_per_sec = (double)(m_model->nsamples()) / elapsedi;

   std::string resumeString = resume ? "Continue from " : std::string();

   std::string phase;
   int i, from;
   if (iteration < 0)
   {
      phase = "Initial";
      i = iteration + 1;
      from = 0;
   }
   else if (iteration < m_config.getBurnin())
   {
      phase = "Burnin";
      i = iteration + 1;
      from = m_config.getBurnin();
   }
   else
   {
      phase = "Sample";
      i = iteration - m_config.getBurnin() + 1;
      from = m_config.getNSamples();
   }

   if (m_config.getVerbose() > 0)
   {
       if (iteration < 0)
       {
           output << " ====== Initial phase ====== " << std::endl;
       }
       else if (iteration < m_config.getBurnin() && iteration == 0)
       {
           output << " ====== Sampling (burning phase) ====== " << std::endl;
       }
       else if (iteration == m_config.getBurnin())
       {
           output << " ====== Burn-in complete, averaging samples ====== " << std::endl;
       }

       output << resumeString
           << phase
           << " "
           << std::setfill(' ') << std::setw(3) << i
           << "/"
           << std::setfill(' ') << std::setw(3) << from
           << ": RMSE: "
           << std::fixed << std::setprecision(4) << m_pred->rmse_avg
           << " (1samp: "
           << std::fixed << std::setprecision(4) << m_pred->rmse_1sample
           << ")";

       if (m_config.getClassify())
       {
           output << " AUC:"
               << std::fixed << std::setprecision(4) << m_pred->auc_avg
               << " (1samp: "
               << std::fixed << std::setprecision(4) << m_pred->auc_1sample
               << ")"
               << std::endl;
       }

       output << "  U:["
           << std::scientific << std::setprecision(2) << snorm0
           << ", "
           << std::scientific << std::setprecision(2) << snorm1
           << "] [took: "
           << std::fixed << std::setprecision(1) << elapsedi
           << "s]"
           << std::endl;

       if (m_config.getVerbose() > 1)
       {
           train_rmse = data()->train_rmse(m_model);
           output << std::fixed << std::setprecision(4) << "  RMSE train: " << train_rmse << std::endl;
           output << "  Priors:" << std::endl;

           for(const auto &p : m_priors)
               p->status(output, "     ");

           output << "  Model:" << std::endl;
           m_model->status(output, "    ");
           output << "  Noise:" << std::endl;
           data()->status(output, "    ");
       }

       if (m_config.getVerbose() > 2)
       {
           output << "  Compute Performance: " << samples_per_sec << " samples/sec, " << nnz_per_sec << " nnz/sec" << std::endl;
       }
   }

   if (m_config.getCsvStatus().size())
   {
      // train_rmse is printed as NAN, unless verbose > 1
      auto f = fopen(m_config.getCsvStatus().c_str(), "a");
      THROWERROR_ASSERT_MSG(f, "Could not open status csv file: " + m_config.getCsvStatus());
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
