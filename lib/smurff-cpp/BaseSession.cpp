#include "BaseSession.h"

#include "ILatentPrior.h"

using namespace smurff;

void BaseSession::step() 
{
   for(auto &p : priors) p->sample_latents();
   data().update(model);
}

std::ostream &BaseSession::info(std::ostream &os, std::string indent) 
{
   os << indent << name << " {\n";
   os << indent << "  Data: {\n";
   data().info(os, indent + "    ");
   os << indent << "  }\n";
   os << indent << "  Model: {\n";
   model.info(os, indent + "    ");
   os << indent << "  }\n";
   os << indent << "  Priors: {\n";
   for( auto &p : priors) p->info(os, indent + "    ");
   os << indent << "  }\n";
   os << indent << "  Result: {\n";
   pred.info(os, indent + "    ", data());
   os << indent << "  }\n";
   return os;
}

void BaseSession::save(std::string prefix, std::string suffix) 
{
   model.save(prefix, suffix);
   pred.save(prefix);
   for(auto &p : priors) 
      p->save(prefix, suffix);
}

void BaseSession::restore(std::string prefix, std::string suffix) 
{
   model.restore(prefix, suffix);
   for(auto &p : priors) 
      p->restore(prefix, suffix);
}
