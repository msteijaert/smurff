#pragma once

#include <string>
#include <vector>
#include <memory>

#include "Data.h"
#include "model.h"
#include "result.h"

namespace smurff {

class ILatentPrior;

class BaseSession  
{
public:
   Model model;
   Result pred;

public:
   std::vector<std::unique_ptr<ILatentPrior> > priors;
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
   template<class Prior>
   Prior& addPrior()
   {
      auto pos = priors.size();
      Prior *p = new Prior(*this, pos);
      priors.push_back(std::unique_ptr<ILatentPrior>(p));
      return *p;
   }

   virtual void step();

   virtual std::ostream &info(std::ostream &, std::string indent);

   void save(std::string prefix, std::string suffix);

   void restore(std::string prefix, std::string suffix);
};

}