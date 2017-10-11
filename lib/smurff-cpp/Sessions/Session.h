#pragma once

#include <iostream>

#include "BaseSession.h"
#include <Configs/Config.h>

namespace smurff {

class Session : public BaseSession 
{
public:
   Config config;
   int iter = -1;

public:
   Session() 
   {
      name = "Session"; 
   }

public:
   void setFromConfig(const Config &);

   // execution of the sampler
public:
   void init();
   void run();
   void step() override;

public:
   std::ostream &info(std::ostream &, std::string indent) override;

 private:
    void save(int isample);
    void printStatus(double elapsedi);
};

}