#pragma once

#include <string>
#include <vector>
#include <memory>

#include <SmurffCpp/DataMatrices/Data.h>
#include <SmurffCpp/Model.h>
#include <SmurffCpp/result.h>

namespace smurff {

class ILatentPrior;

class BaseSession  
{
protected:
   Model m_model;
   Result m_pred;

protected:
   std::vector<std::shared_ptr<ILatentPrior> > m_priors;
   std::string name;

protected:
   bool is_init = false;

   //train data
   std::shared_ptr<Data> data_ptr;

public:
   virtual ~BaseSession() {}

public:
   std::shared_ptr<Data> data() const
   { 
      assert(data_ptr); 
      return data_ptr; 
   }

   const Model& model() const
   {
      return m_model;
   }

   Model& model()
   {
      return m_model;
   }

public:
   void addPrior(std::shared_ptr<ILatentPrior> prior);

protected:
   virtual void step();

public:
   virtual std::ostream &info(std::ostream &, std::string indent);

   void save(std::string prefix, std::string suffix);

   void restore(std::string prefix, std::string suffix);
};

}