#pragma once

#include <string>
#include <vector>
#include <memory>

#include <SmurffCpp/DataMatrices/Data.h>
#include <SmurffCpp/model.h>
#include <SmurffCpp/result.h>

namespace smurff {

class ILatentPrior;

class BaseSession  
{
public:
   Model model;
   Result pred;

protected:
   std::vector<std::shared_ptr<ILatentPrior> > m_priors;

   std::string name;

protected:
   bool is_init = false;
   std::unique_ptr<Data> data_ptr;

public:
   virtual ~BaseSession() {}

public:
   Data& data() 
   { 
      assert(data_ptr); 
      return *data_ptr; 
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