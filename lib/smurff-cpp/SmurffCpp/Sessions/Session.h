#pragma once

#include <iostream>
#include <memory>

#include <SmurffCpp/Utils/Error.h>
#include <SmurffCpp/Configs/Config.h>
#include <SmurffCpp/Priors/IPriorFactory.h>
#include <SmurffCpp/Utils/RootFile.h>
#include <SmurffCpp/StatusItem.h>
#include <SmurffCpp/Sessions/ISession.h>

namespace smurff {

class SessionFactory;

class Session : public ISession, public std::enable_shared_from_this<Session>
{
   //only session factory should call setFromConfig
   friend class SessionFactory;

protected:
   std::shared_ptr<Model> m_model;
   std::shared_ptr<Result> m_pred;

protected:
   std::vector<std::shared_ptr<ILatentPrior> > m_priors;
   std::string name;

protected:
   bool is_init = false;

   //train data
   std::shared_ptr<Data> data_ptr;

private:
   std::shared_ptr<RootFile> m_rootFile;

protected:
   Config m_config;

private:
   int m_iter = -1; //index of step iteration
   double m_secs_per_iter = .0; //time in seconds for last_iter
   double m_secs_total = .0; //time in seconds for last_iter
   double m_lastCheckpointTime;
   int m_lastCheckpointIter;

public:
   bool inBurninPhase() const { return m_iter < m_config.getBurnin(); }
   bool inSamplingPhase() const { return !inBurninPhase(); }
   bool finalSample() const { return m_iter == (m_config.getNSamples() + m_config.getBurnin()); }

protected:
   Session();

public:
   void addPrior(std::shared_ptr<ILatentPrior> prior);

public:
   std::shared_ptr<Result> getResult() const override;
public:
   void fromRootPath(std::string rootPath);
   void fromConfig(const Config& cfg);

protected:
   void setFromBase();

   // execution of the sampler
public:
   void run() override;

protected:
   void init() override;

public:
   bool step() override;

public:
   std::ostream &info(std::ostream &, std::string indent) const override;

private:
   //save current iteration
   void save(int iteration);

   void saveInternal(std::shared_ptr<StepFile> stepFile);

   //restore last iteration
   bool restore(int& iteration);

private:
   void printStatus(std::ostream& output, bool resume = false);

public:
   std::shared_ptr<StatusItem> getStatus() const override;

private:
   void initRng();

public:
   virtual std::shared_ptr<IPriorFactory> create_prior_factory() const;

   std::shared_ptr<RootFile> getRootFile() const override
   {
       THROWERROR_ASSERT_MSG(m_rootFile, "No root file found. Did you save any models?");
       return m_rootFile;
   }

public:
   const Config& getConfig()
   {
      return m_config;
   }

   Data &data() const
   {
      THROWERROR_ASSERT(data_ptr != 0);
      return *data_ptr;
   }

   const Model& model() const
   {
      return *m_model;
   }

   Model& model()
   {
      return *m_model;
   }


};

} // end namespace
