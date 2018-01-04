#include "MPIPriorFactory.h"

#include <SmurffCpp/Configs/Config.h>
#include <SmurffCpp/Priors/ILatentPrior.h>

using namespace smurff;

std::shared_ptr<ILatentPrior> MPIPriorFactory::create_prior(std::shared_ptr<Session> session, int mode)
{
   PriorTypes pt = session->config.getPriorTypes().at(mode);

   if(pt == PriorTypes::macau)
   {
   }
   else
   {
      return PriorFactory::create_prior(session, mode);
   }
}
