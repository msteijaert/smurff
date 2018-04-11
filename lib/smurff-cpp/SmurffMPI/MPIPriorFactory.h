#pragma once

#include <memory>

#include <SmurffCpp/Priors/PriorFactory.h>

namespace smurff {

   class ILatentPrior;
   class Session;

   class MPIPriorFactory : public PriorFactory
   {
   public:
      std::shared_ptr<ILatentPrior> create_macau_prior(std::shared_ptr<Session> session, PriorTypes prior_type,
                                                       const std::vector<std::shared_ptr<ISideInfo> >& side_infos,
                                                       const std::vector<std::shared_ptr<SideInfoConfig> >& config_items);

      std::shared_ptr<ILatentPrior> create_prior(std::shared_ptr<Session> session, int mode) override;
   };
}
