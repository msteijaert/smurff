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

public:
   Config config;
   int iter = -1; //index of step iteration

protected:
   Session()
   {
      name = "Session";
   }

protected:
   void setFromRootPath(std::string rootPath);
   void setFromConfig(const Config& cfg);
   void setFromBase();

   // execution of the sampler
public:
   void run() override;

protected:
   void init() override;

public:
   void step() override;

public:
   std::ostream &info(std::ostream &, std::string indent) override;

private:
   //save iteration
   void save(int isample);
   void restore();
   void printStatus(double elapsedi);

public:
   virtual std::shared_ptr<IPriorFactory> create_prior_factory() const;
};

}