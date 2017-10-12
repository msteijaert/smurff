#pragma once

#include <string>
#include <iostream>

namespace smurff {

   class Data;
   struct SubModel;

   // interface
   class INoiseModel
   {
      // Only Data can call init and update methods
      friend class Data;

   public:
      INoiseModel() {}

   public:
      virtual ~INoiseModel() {}

   protected:
      virtual void init(const Data* data) {}
      virtual void update(const Data* data, const SubModel & model) {}

   public:
      virtual std::ostream &info(std::ostream &os, std::string indent)   = 0;
      virtual std::string getStatus()  = 0;

      virtual double getAlpha() = 0;
   };
}
