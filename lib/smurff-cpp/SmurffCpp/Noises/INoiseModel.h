#pragma once

#include <string>
#include <iostream>

#include <SmurffCpp/Utils/PVec.hpp>

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

   private:
      const Data* m_data;
   
   protected:
      INoiseModel() : m_data(0) {}

   public:
      virtual ~INoiseModel()  {}

   protected:
      virtual void init(const Data* data) { m_data = data;}
      virtual void update(const SubModel & model) {}
      const Data& data() const { return *m_data; }

   public:
      virtual std::ostream &info(std::ostream &os, std::string indent)   = 0;
      virtual std::string getStatus()  = 0;

      virtual double getAlpha(const SubModel& model, const PVec<> &pos, double val) = 0;
   };
}
