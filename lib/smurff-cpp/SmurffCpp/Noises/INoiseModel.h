#pragma once

#include <string>
#include <iostream>

namespace smurff {

   class Data;
   class SubModel;
   class NoiseFactory;

   // interface
   class INoiseModel
   {
      friend class NoiseFactory;
      
      // Only Data can call init and update methods
      friend class Data;

   protected:
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
