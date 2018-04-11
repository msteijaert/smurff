#include "MPIPriorFactory.h"

#include <SmurffCpp/Configs/Config.h>
#include <SmurffCpp/Priors/ILatentPrior.h>
#include <SmurffMPI/MPIPriorFactory.h>
#include <SmurffMPI/MPIMacauPrior.h>

using namespace smurff;

std::shared_ptr<ILatentPrior> MPIPriorFactory::create_macau_prior(std::shared_ptr<Session> session, PriorTypes prior_type, 
                                                                  const std::vector<std::shared_ptr<ISideInfo> >& side_infos,
                                                                  const std::vector<std::shared_ptr<SideInfoConfig> >& config_items)
{
   return PriorFactory::create_macau_prior<MPIMacauPrior>(session, side_infos, config_items);
}

std::shared_ptr<ILatentPrior> MPIPriorFactory::create_prior(std::shared_ptr<Session> session, int mode)
{
   PriorTypes pt = session->getConfig().getPriorTypes().at(mode);

   if(pt == PriorTypes::macau)
   {
      return PriorFactory::create_macau_prior<MPIPriorFactory>(session, mode, pt, session->getConfig().getSideInfoConfigs(mode));
   }
   else
   {
      return PriorFactory::create_prior(session, mode);
   }
}
