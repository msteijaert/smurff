#pragma once

#include <iostream>
#include <memory>

#include "BaseSession.h"
#include <SmurffCpp/Configs/Config.h>
#include <SmurffCpp/Priors/IPriorFactory.h>
#include <SmurffCpp/Utils/RootFile.h>

namespace smurff {

class SessionFactory;

class Session : public BaseSession, public std::enable_shared_from_this<Session>
{
   //only session factory should call setFromConfig
   friend class SessionFactory;

private:
   std::shared_ptr<RootFile> m_rootFile;

protected:
   Config m_config;

private:
   int m_iter = -1; //index of step iteration
   double m_lastCheckpointTime;
   int m_lastCheckpointIter;

protected:
   Session()
   {
      name = "Session";
   }

protected:
   void setRestoreFromRootPath(std::string rootPath);
   void setRestoreFromConfig(const Config& cfg, std::string rootPath);
   void setCreateFromConfig(const Config& cfg);
   void setFromBase();

   // execution of the sampler
public:
   void run() override;

protected:
   void init() override;

public:
   bool step() override;

public:
   std::ostream &info(std::ostream &, std::string indent) override;

private:
   //save current iteration
   void save(int iteration);

   void saveInternal(std::shared_ptr<StepFile> stepFile);

   //restore last iteration
   bool restore(int& iteration);

private:
   void printStatus(std::ostream& output, double elapsedi, bool resume, int isample);

private:
   void initRng();

public:
   virtual std::shared_ptr<IPriorFactory> create_prior_factory() const;

public:
   const Config& getConfig()
   {
      return m_config;
   }
};

}
