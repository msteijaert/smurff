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
    std::shared_ptr<ILatentPrior> create_macau_prior(std::shared_ptr<Session> session,
                                                     const std::vector<std::shared_ptr<typename MacauPrior::SideInfo> >& side_infos,
                                                     const std::vector<std::shared_ptr<MacauPriorConfigItem> >& config_items);

    template<class SideInfo>
    std::shared_ptr<ILatentPrior> create_macau_prior(std::shared_ptr<Session> session, PriorTypes prior_type, 
                                                     const std::vector<std::shared_ptr<SideInfo> >& side_infos,
                                                     const std::vector<std::shared_ptr<MacauPriorConfigItem> >& config_items);

    template<class Factory>
    std::shared_ptr<ILatentPrior> create_macau_prior(std::shared_ptr<Session> session, int mode, PriorTypes prior_type, const std::shared_ptr<MacauPriorConfig>& priorConfig);

    std::shared_ptr<ILatentPrior> create_prior(std::shared_ptr<Session> session, int mode) override;
};

//-------

template<class MacauPrior>
std::shared_ptr<ILatentPrior> PriorFactory::create_macau_prior(std::shared_ptr<Session> session,
                                                               const std::vector<std::shared_ptr<typename MacauPrior::SideInfo> >& side_infos,
                                                               const std::vector<std::shared_ptr<MacauPriorConfigItem> >& config_items)
{
   THROWERROR_ASSERT(side_infos.size() == config_items.size());

   std::shared_ptr<MacauPrior> prior(new MacauPrior(session, -1));

   for (std::size_t i = 0; i < side_infos.size(); i++)
   {
      const auto& side_info = side_infos.at(i);
      const auto& config_item = config_items.at(i);
      const auto& side_info_config = config_item->getSideInfo();
      const auto& noise_config = side_info_config->getNoiseConfig();

      switch (noise_config.getNoiseType())
      {
      case NoiseTypes::fixed:
         {
            prior->addSideInfo(side_info, noise_config.getPrecision(), config_item->getTol(), config_item->getDirect(), false);
         }
         break;
      case NoiseTypes::adaptive:
         {
            prior->addSideInfo(side_info, noise_config.getPrecision(), config_item->getTol(), config_item->getDirect(), true);
         }
         break;
      default:
         {
            THROWERROR("Unexpected noise type " + smurff::noiseTypeToString(noise_config.getNoiseType()) + "specified for macau prior");
         }
      }
   }

   return prior;
}

template<class SideInfo>
std::shared_ptr<ILatentPrior> PriorFactory::create_macau_prior(std::shared_ptr<Session> session, PriorTypes prior_type, 
                                                               const std::vector<std::shared_ptr<SideInfo> >& side_infos,
                                                               const std::vector<std::shared_ptr<MacauPriorConfigItem> >& config_items)
{
   if(prior_type == PriorTypes::macau || prior_type == PriorTypes::default_prior)
   {
      return create_macau_prior<MacauPrior<SideInfo>>(session, side_infos, config_items);
   }
   else if(prior_type == PriorTypes::macauone)
   {
      return create_macau_prior<MacauOnePrior<SideInfo>>(session, side_infos, config_items);
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
   //at the moment different side info types can not be stored into single vector - we need a proper interface
   //this is the reason why we take only first side info.
   //however code is written so that it uses vectors to make it easy to fix in the future

   auto& configItem = priorConfig->getConfigItems().front();
   const auto& sideinfoConfig = configItem->getSideInfo();

   if (sideinfoConfig->isBinary())
   {
      std::vector<std::shared_ptr<SparseFeat> > side_infos;
      side_infos.push_back(side_info_config_to_sparse_binary_features(sideinfoConfig, mode));

      std::vector<std::shared_ptr<MacauPriorConfigItem> > config_items;
      config_items.push_back(configItem);

      return subFactory.create_macau_prior(session, prior_type, side_infos, config_items);
   }
   else if (sideinfoConfig->isDense())
   {
      std::vector<std::shared_ptr<Eigen::MatrixXd> > side_infos;
      side_infos.push_back(side_info_config_to_dense_features(sideinfoConfig, mode));

      std::vector<std::shared_ptr<MacauPriorConfigItem> > config_items;
      config_items.push_back(configItem);

      return subFactory.create_macau_prior(session, prior_type, side_infos, config_items);
   }
   else
   {
      std::vector<std::shared_ptr<SparseDoubleFeat> > side_infos;
      side_infos.push_back(side_info_config_to_sparse_features(sideinfoConfig, mode));

      std::vector<std::shared_ptr<MacauPriorConfigItem> > config_items;
      config_items.push_back(configItem);

      return subFactory.create_macau_prior(session, prior_type, side_infos, config_items);
   }
}
}
