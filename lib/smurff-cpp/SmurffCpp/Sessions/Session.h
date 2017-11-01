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
   int iter = -1;

public:
   Session() 
   {
      name = "Session"; 
   }

protected:
   void setFromConfig(const Config& cfg);

   // execution of the sampler
public:
   void run();

   //AGE: I guess these methods should be internal and called only from run ?
protected:
   void init();
   void step() override;

public:
   std::ostream &info(std::ostream &, std::string indent) override;

 private:
    void save(int isample);
    void printStatus(double elapsedi);
};

}