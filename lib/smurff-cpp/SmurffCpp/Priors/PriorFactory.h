#pragma once

#include <SmurffCpp/Priors/ILatentPrior.h>
#include <SmurffCpp/Sessions/Session.h>
#include <SmurffCpp/Configs/MatrixConfig.h>

namespace smurff {

class PriorFactory
{
public:
   static std::shared_ptr<ILatentPrior> create_prior(std::shared_ptr<Session> session, int mode);
};

}