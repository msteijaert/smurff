#pragma once

#include <vector>
#include <memory>

#include <SmurffCpp/ResultItem.h>
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
      virtual bool step() = 0;
      virtual bool interrupted() { return false; }
      virtual void init() = 0;
      virtual std::shared_ptr<std::vector<ResultItem> > getResult() = 0;
      virtual MatrixConfig getSample(int mode) = 0;
      virtual double getRmseAvg() = 0;

   public:
      virtual std::ostream &info(std::ostream &, std::string indent) = 0;
   };

}
