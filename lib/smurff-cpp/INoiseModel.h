#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <memory>
#include <iomanip>
#include "utils.h"

namespace smurff {

   // AGE: I dont like this cross reference between Data and INoiseModel. Need to think how we can eliminate it.
   // AGE: Same applies to SubModel
   struct Data;
   struct SubModel;

   // interface
   class INoiseModel 
   {
   protected:
      Data* data;

   public:
      INoiseModel(Data* p)
         : data(p) {}

   public:
      virtual ~INoiseModel() {}

   public:
      virtual void init()  = 0;
      virtual void update(const SubModel &)  = 0;

      virtual std::ostream &info(std::ostream &os, std::string indent)   = 0;
      virtual std::string getStatus()  = 0;

      virtual double getAlpha() = 0;
   };
}
