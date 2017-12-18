#pragma once

#include <iostream>
#include <memory>

#include "BaseSession.h"
#include <SmurffCpp/Configs/Config.h>

namespace smurff {

class SessionFactory;

class Session : public BaseSession, public std::enable_shared_from_this<Session>
{
   //only session factory should call setFromConfig
   friend class SessionFactory;

public:
   Config config;
   int iter = -1; //index of step iteration

protected:
   Session()
   {
      name = "Session";
   }

protected:
   void setFromConfig(const Config& cfg);

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
   void printStatus(double elapsedi);
};

}