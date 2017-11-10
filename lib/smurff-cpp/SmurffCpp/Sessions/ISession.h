#pragma once

#include <SmurffCpp/Configs/MatrixConfig.h>

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
      virtual MatrixConfig getResult() = 0;
      virtual MatrixConfig getSample(int dim) = 0;
   };

}