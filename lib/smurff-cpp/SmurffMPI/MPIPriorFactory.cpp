#include "MPIPriorFactory.h"

#include <SmurffCpp/Configs/Config.h>
#include <SmurffCpp/Priors/ILatentPrior.h>
#include <SmurffMPI/MPIPriorFactory.h>
#include <SmurffMPI/MPIMacauPrior.hpp>

using namespace smurff;

template<class SideInfo>
std::shared_ptr<ILatentPrior> MPIPriorFactory::create_macau_prior(std::shared_ptr<Session> session, PriorTypes prior_type, std::shared_ptr<SideInfo> side_info)
{
   return PriorFactory::create_macau_prior<MPIMacauPrior<SideInfo>>(session, side_info);
}

std::shared_ptr<ILatentPrior> MPIPriorFactory::create_prior(std::shared_ptr<Session> session, int mode)
{
   PriorTypes pt = session->config.getPriorTypes().at(mode);

   if(pt == PriorTypes::macau)
   {
      return PriorFactory::create_macau_prior<MPIPriorFactory>(session, mode, pt, session->config.getSideInfo().at(mode));
   }
   else
   {
      return PriorFactory::create_prior(session, mode);
   }
}
