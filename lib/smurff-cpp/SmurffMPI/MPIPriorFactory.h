#pragma once

#include <memory>

#include <SmurffCpp/Priors/PriorFactory.h>

namespace smurff {
   
   class ILatentPrior;
   class Session;

   class MPIPriorFactory : public PriorFactory
   {
   public:
      std::shared_ptr<ILatentPrior> create_prior(std::shared_ptr<Session> session, int mode) override;
   };
}