#pragma once

#include <vector>
#include <memory>

#include <SmurffCpp/StatusItem.h>
#include <SmurffCpp/ResultItem.h>
#include <SmurffCpp/Configs/MatrixConfig.h>

namespace smurff {
   class RootFile;
   class Result;

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
      virtual std::shared_ptr<Result> getResult() const = 0;
      virtual std::shared_ptr<RootFile> getRootFile() const = 0;

      float getRmseAvg() { return getStatus()->rmse_avg; }
      const std::vector<ResultItem> & getResultItems() const;

    public:
      virtual std::ostream &info(std::ostream &, std::string indent) const = 0;
      std::string infoAsString() 
      {
          std::ostringstream ss;
          info(ss, "");
          return ss.str();
      }
   };

}
