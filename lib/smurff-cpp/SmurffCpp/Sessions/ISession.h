#pragma once

#include <vector>
#include <memory>

#include <SmurffCpp/ResultItem.h>
#include <SmurffCpp/StatusItem.h>
#include <SmurffCpp/Configs/MatrixConfig.h>

namespace smurff {
   class RootFile;

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

      virtual std::shared_ptr<StatusItem> getStatus() const = 0;
      virtual std::shared_ptr<std::vector<ResultItem> > getResult() const = 0;
      virtual MatrixConfig getSample(int mode) const = 0;

      virtual std::shared_ptr<RootFile> getRootFile() const = 0;

      double getRmseAvg() { return getStatus()->rmse_avg; }

    public:
      virtual std::ostream &info(std::ostream &, std::string indent) = 0;
      std::string infoAsString() 
      {
          std::ostringstream ss;
          info(ss, "");
          return ss.str();
      }
   };

}
