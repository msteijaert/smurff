#pragma once

#include <vector>

#include <SmurffCpp/Configs/MatrixConfig.h>
#include <SmurffCpp/result.h>

namespace smurff {

   class ISession
   {
   protected:
      ISession(){};

   public:
      virtual ~ISession(){}

   public:
      virtual void run() = 0;
      virtual void step() = 0;
      virtual void init() = 0;
      virtual std::vector<Result::Item> getResult() = 0;
      virtual MatrixConfig getSample(int mode) = 0;
   };

}