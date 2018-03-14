#pragma once

#include <SmurffCpp/Priors/IPriorFactory.h>
#include <SmurffCpp/Priors/ILatentPrior.h>
#include <SmurffCpp/Priors/MacauOnePrior.hpp>
#include <SmurffCpp/Priors/MacauPrior.hpp>
#include <SmurffCpp/Sessions/Session.h>
#include <SmurffCpp/Configs/MatrixConfig.h>
#include <SmurffCpp/Configs/MacauPriorConfig.h>

namespace smurff {

class PriorFactory : public IPriorFactory
{
private:

    std::shared_ptr<Eigen::MatrixXd> side_info_config_to_dense_features(std::shared_ptr<MatrixConfig> sideinfoConfig, int mode);
    std::shared_ptr<SparseFeat> side_info_config_to_sparse_binary_features(std::shared_ptr<MatrixConfig> sideinfoConfig, int mode);
    std::shared_ptr<SparseDoubleFeat> side_info_config_to_sparse_features(std::shared_ptr<MatrixConfig> sideinfoConfig, int mode);

public:
    template<class MacauPrior>
    std::shared_ptr<ILatentPrior> create_macau_prior(std::shared_ptr<Session> session, bool direct, std::shared_ptr<typename MacauPrior::SideInfo> side_info,
                                                     const std::vector<std::shared_ptr<MacauPriorConfigItem> >& config_items);

    template<class SideInfo>
    std::shared_ptr<ILatentPrior> create_macau_prior(std::shared_ptr<Session> session, PriorTypes prior_type, bool direct, std::shared_ptr<SideInfo> side_info,
                                                     const std::vector<std::shared_ptr<MacauPriorConfigItem> >& config_items);

    template<class Factory>
    std::shared_ptr<ILatentPrior> create_macau_prior(std::shared_ptr<Session> session, int mode, PriorTypes prior_type, const std::shared_ptr<MacauPriorConfig>& priorConfig);

    std::shared_ptr<ILatentPrior> create_prior(std::shared_ptr<Session> session, int mode) override;
};

//-------

template<class MacauPrior>
std::shared_ptr<ILatentPrior> PriorFactory::create_macau_prior(std::shared_ptr<Session> session, bool direct, std::shared_ptr<typename MacauPrior::SideInfo> side_info,
                                                               const std::vector<std::shared_ptr<MacauPriorConfigItem> >& config_items)
{
    std::shared_ptr<MacauPrior> prior(new MacauPrior(session, -1));

    prior->addSideInfo(side_info, direct);
    prior->setLambdaBetaValues(config_items);
    prior->setTolValues(config_items);

    prior->setEnableLambdaBetaSampling(session->getConfig().getEnableLambdaBetaSampling());
    
    return prior;
}

template<class SideInfo>
std::shared_ptr<ILatentPrior> PriorFactory::create_macau_prior(std::shared_ptr<Session> session, PriorTypes prior_type, bool direct, std::shared_ptr<SideInfo> side_info,
                                                               const std::vector<std::shared_ptr<MacauPriorConfigItem> >& config_items)
{
   if(prior_type == PriorTypes::macau || prior_type == PriorTypes::default_prior)
   {
      return create_macau_prior<MacauPrior<SideInfo>>(session, direct, side_info, config_items);
   }
   else if(prior_type == PriorTypes::macauone)
   {
      return create_macau_prior<MacauOnePrior<SideInfo>>(session, direct, side_info, config_items);
   }
   else
   {
      THROWERROR("Unknown prior with side info: " + priorTypeToString(prior_type));
   }
}

//mode - 0 (row), 1 (col)
//vsideinfo - vector of side feature configs (row or col)
template<class Factory>
std::shared_ptr<ILatentPrior> PriorFactory::create_macau_prior(std::shared_ptr<Session> session, int mode, PriorTypes prior_type, const std::shared_ptr<MacauPriorConfig>& priorConfig)
{
   Factory &subFactory = dynamic_cast<Factory &>(*this);

   if(!priorConfig)
   {
      THROWERROR("Side info should always present for macau prior");
   }

   //FIXME: 
   //there is a problem - do we want different types of side infos for one prior?
   //if yes then we can not use current implementation of MacauPrior for example
   //if we want to store each side info separately since type of sideinfo is specified as template argument
   //we can try to convert side info to Data and finally get rid of legacy classes
   //or we can try to merge HERE all side info into single matrix and store it into macau prior

   auto& configItem = priorConfig->getConfigItems().front();
   auto& sideinfoConfig = configItem->getSideInfo();

   if (sideinfoConfig->isBinary())
   {
      std::shared_ptr<SparseFeat> sideInfo = side_info_config_to_sparse_binary_features(sideinfoConfig, mode);
      return subFactory.create_macau_prior(session, prior_type, configItem->getDirect(), sideInfo, priorConfig->getConfigItems());
   }
   else if (sideinfoConfig->isDense())
   {
      std::shared_ptr<Eigen::MatrixXd> sideInfo = side_info_config_to_dense_features(sideinfoConfig, mode);
      return subFactory.create_macau_prior(session, prior_type, configItem->getDirect(), sideInfo, priorConfig->getConfigItems());
   }
   else
   {
      std::shared_ptr<SparseDoubleFeat> sideInfo = side_info_config_to_sparse_features(sideinfoConfig, mode);
      return subFactory.create_macau_prior(session, prior_type, configItem->getDirect(), sideInfo, priorConfig->getConfigItems());
   }
}
}
