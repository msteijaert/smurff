#include "Session.h"

#include <fstream>
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
#include <SmurffCpp/StatusItem.h>

using namespace smurff;

void Session::setRestoreFromRootPath(std::string rootPath)
{
   // open root file
   m_rootFile = std::make_shared<RootFile>(rootPath);

   //restore config
   m_rootFile->restoreConfig(m_config);

   m_config.validate();

   //base functionality
   setFromBase();
}

void Session::setRestoreFromConfig(const Config& cfg, std::string rootPath)
{
   cfg.validate();

   // assign config
   m_config = cfg;

   // open root file
   m_rootFile = std::make_shared<RootFile>(rootPath);

   //base functionality
   setFromBase();
}

void Session::setCreateFromConfig(const Config& cfg)
{
   cfg.validate();

   // assign config
   m_config = cfg;

   if (m_config.getSaveFreq() || m_config.getCheckpointFreq())
   {

       // create root file
       m_rootFile = std::make_shared<RootFile>(m_config.getSavePrefix(), m_config.getSaveExtension());

       //save config
       m_rootFile->saveConfig(m_config);

       //flush record about options.ini
       m_rootFile->flushLast();
   }

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
   //init omp
   threads_init(m_config.getVerbose());

   //initialize random generator
   initRng();

   //initialize train matrix (centring and noise model)
   data().init();

   //initialize model (samples)
   model().init(m_config.getNumLatent(), data().dim(), m_config.getModelInitType());

   //initialize priors
   for(auto &p : m_priors)
      p->init();

   //write header to status file
   if (m_config.getCsvStatus().size())
   {
      auto f = std::ofstream(m_config.getCsvStatus(), std::ofstream::out);
      THROWERROR_ASSERT_MSG(f, "Could not open status csv file: " + m_config.getCsvStatus());
      f << StatusItem::getCsvHeader() << std::endl;
   }

   //write info to console
   if (m_config.getVerbose())
      info(std::cout, "");

   //restore session (model, priors)
   bool resume = restore(m_iter);

   //print session status to console
   if (m_config.getVerbose())
   {
      printStatus(std::cout, resume);
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

      m_secs_per_iter = endi - starti;

      printStatus(std::cout);

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
   std::shared_ptr<StepFile> stepFile = nullptr;
   if (m_rootFile)
   {
       stepFile = m_rootFile->openLastStepFile();
   }

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

std::shared_ptr<StatusItem> Session::getStatus() const
{
   std::shared_ptr<StatusItem> ret = std::make_shared<StatusItem>();

   if (m_iter < 0)
   {
      ret->phase = "Initial";
      ret->iter = m_iter + 1;
      ret->phase_iter = 0;
   }
   else if (m_iter < m_config.getBurnin())
   {
      ret->phase = "Burnin";
      ret->iter = m_iter + 1;
      ret->phase_iter = m_config.getBurnin();
   }
   else
   {
      ret->phase = "Sample";
      ret->iter = m_iter - m_config.getBurnin() + 1;
      ret->phase_iter = m_config.getNSamples();
   }

   for(int i=0; i<model().nmodes(); ++i)
   {
       ret->model_norms.push_back(model().U(i).norm());
   }
    
   ret->nnz_per_sec =  (double)(data().nnz()) / m_iter;
   ret->samples_per_sec = (double)(model().nsamples()) / m_iter;
   ret->iter = m_iter + 1;
   ret->train_rmse = data().train_rmse(model());

    ret->rmse_avg = m_pred->rmse_avg;
    ret->rmse_1sample = m_pred->rmse_1sample;

    ret->auc_avg = m_pred->auc_avg;
    ret->auc_1sample = m_pred->auc_1sample;

    return ret;
}

void Session::printStatus(std::ostream& output, bool resume)
{
   if(!m_config.getVerbose() &&
      !m_config.getCsvStatus().size())
      return;

   auto status_item = getStatus();

   std::string resumeString = resume ? "Continue from " : std::string();

   if (m_config.getVerbose() > 0)
   {
       if (m_iter < 0)
       {
           output << " ====== Initial phase ====== " << std::endl;
       }
       else if (m_iter < m_config.getBurnin() && m_iter == 0)
       {
           output << " ====== Sampling (burning phase) ====== " << std::endl;
       }
       else if (m_iter == m_config.getBurnin())
       {
           output << " ====== Burn-in complete, averaging samples ====== " << std::endl;
       }

       output << resumeString
           << status_item->phase
           << " "
           << std::setfill(' ') << std::setw(3) << status_item->iter
           << "/"
           << std::setfill(' ') << std::setw(3) << status_item->phase_iter
           << ": RMSE: "
           << std::fixed << std::setprecision(4) << status_item->rmse_avg
           << " (1samp: "
           << std::fixed << std::setprecision(4) << status_item->rmse_1sample
           << ")";

       if (m_config.getClassify())
       {
           output << " AUC:"
               << std::fixed << std::setprecision(4) << status_item->auc_avg
               << " (1samp: "
               << std::fixed << std::setprecision(4) << status_item->auc_1sample
               << ")"
               << std::endl;
       }

       output << "  U:[";
       for(double n: status_item->model_norms)
       {
           output << std::scientific << std::setprecision(2) << n << ", ";
       }
       output << "] [took: "
           << std::fixed << std::setprecision(1) << status_item->elapsed_iter
           << "s]"
           << std::endl;

       if (m_config.getVerbose() > 1)
       {
           output << std::fixed << std::setprecision(4) << "  RMSE train: " << status_item->train_rmse << std::endl;
           output << "  Priors:" << std::endl;

           for(const auto &p : m_priors)
               p->status(output, "     ");

           output << "  Model:" << std::endl;
           model().status(output, "    ");
           output << "  Noise:" << std::endl;
           data().status(output, "    ");
       }

       if (m_config.getVerbose() > 2)
       {
           output << "  Compute Performance: " << status_item->samples_per_sec << " samples/sec, " << status_item->nnz_per_sec << " nnz/sec" << std::endl;
       }
   }

   if (m_config.getCsvStatus().size())
   {
      auto f = std::ofstream(m_config.getCsvStatus(), std::ofstream::out | std::ofstream::app);
      THROWERROR_ASSERT_MSG(f, "Could not open status csv file: " + m_config.getCsvStatus());
      f << status_item->asCsvString() << std::endl;
   }
}

std::string StatusItem::getCsvHeader()
{
   return "phase;iter;phase_len;rmse_avg;rmse_1samp;train_rmse;auc_avg;auc_1samp;elapsed";
}

std::string StatusItem::asCsvString() const
{
    char ret[1024];
    snprintf(ret, 1024, "%s;%d;%d;%.4f;%.4f;%.4f;%.4f;:%.4f;%0.1f",
                  phase.c_str(), iter, phase_iter, rmse_avg, rmse_1sample, train_rmse,
                  auc_1sample, auc_avg, elapsed_iter);

    return ret;
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
