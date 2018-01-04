#pragma once

#include <SmurffCpp/Priors/IPriorFactory.h>
#include <SmurffCpp/Priors/ILatentPrior.h>
#include <SmurffCpp/Sessions/Session.h>
#include <SmurffCpp/Configs/MatrixConfig.h>

namespace smurff {

class PriorFactory : public IPriorFactory
{
private:
    template<class MacauPrior>
    std::shared_ptr<ILatentPrior> create_macau_prior(std::shared_ptr<Session> session, std::shared_ptr<typename MacauPrior::SideInfo> side_info);

    template<class SideInfo>
    std::shared_ptr<ILatentPrior> create_macau_prior(std::shared_ptr<Session> session, PriorTypes prior_type, std::shared_ptr<SideInfo> side_info);

    std::shared_ptr<ILatentPrior> create_macau_prior(std::shared_ptr<Session> session, int mode, PriorTypes prior_type, const std::shared_ptr<MatrixConfig>& sideinfoConfig);

public:
   std::shared_ptr<ILatentPrior> create_prior(std::shared_ptr<Session> session, int mode) override;
};

}
