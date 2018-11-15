#pragma once

#include <SmurffCpp/Priors/IPriorFactory.h>
#include <SmurffCpp/Priors/ILatentPrior.h>
#include <SmurffCpp/Priors/MacauOnePrior.h>
#include <SmurffCpp/Priors/MacauPrior.h>
#include <SmurffCpp/Sessions/Session.h>
#include <SmurffCpp/Configs/MatrixConfig.h>
#include <SmurffCpp/Configs/SideInfoConfig.h>

#include <SmurffCpp/SideInfo/DenseSideInfo.h>
#include <SmurffCpp/SideInfo/SparseSideInfo.h>

namespace smurff {

class PriorFactory : public IPriorFactory
{
public:
    template<class MacauPrior>
    std::shared_ptr<ILatentPrior> create_macau_prior(std::shared_ptr<Session> session,
                                                     const std::vector<std::shared_ptr<ISideInfo> >& side_infos,
                                                     const std::vector<std::shared_ptr<SideInfoConfig> >& config_items);

    std::shared_ptr<ILatentPrior> create_macau_prior(std::shared_ptr<Session> session, PriorTypes prior_type, 
                                                     const std::vector<std::shared_ptr<ISideInfo> >& side_infos,
                                                     const std::vector<std::shared_ptr<SideInfoConfig> >& config_items);

    template<class Factory>
    std::shared_ptr<ILatentPrior> create_macau_prior(std::shared_ptr<Session> session, int mode, PriorTypes prior_type,
            const std::vector<std::shared_ptr<SideInfoConfig> >& config_items);

    std::shared_ptr<ILatentPrior> create_prior(std::shared_ptr<Session> session, int mode) override;
};

//-------

template<class MacauPrior>
std::shared_ptr<ILatentPrior> PriorFactory::create_macau_prior(std::shared_ptr<Session> session,
                                                               const std::vector<std::shared_ptr<ISideInfo> >& side_infos,
                                                               const std::vector<std::shared_ptr<SideInfoConfig> >& config_items)
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
            prior->addSideInfo(side_info, noise_config.getPrecision(), config_item->getTol(), config_item->getDirect(), false, config_item->getThrowOnCholeskyError());
         }
         break;
      case NoiseTypes::adaptive: // deprecated!
      case NoiseTypes::sampled:
         {
            prior->addSideInfo(side_info, noise_config.getPrecision(), config_item->getTol(), config_item->getDirect(), true, config_item->getThrowOnCholeskyError());
         }
         break;
      default:
         {
            THROWERROR("Unexpected noise type " + smurff::noiseTypeToString(noise_config.getNoiseType()) + " specified for macau prior. Allowed are: fixed and sampled noise.");
         }
      }
   }

   return prior;
}

//mode - 0 (row), 1 (col)
//vsideinfo - vector of side feature configs (row or col)
template<class Factory>
std::shared_ptr<ILatentPrior> PriorFactory::create_macau_prior(std::shared_ptr<Session> session, int mode, PriorTypes prior_type,
        const std::vector<std::shared_ptr<SideInfoConfig> >& config_items)
{
   Factory &subFactory = dynamic_cast<Factory &>(*this);

   std::vector<std::shared_ptr<ISideInfo> > side_infos;

   for (auto& item : config_items)
   {
      const auto &sideinfoConfig = item->getSideInfo();

      if (sideinfoConfig->isDense())
      {
         side_infos.push_back(std::make_shared<DenseSideInfo>(sideinfoConfig));
      }
      else
      {
         side_infos.push_back(std::make_shared<SparseSideInfo>(sideinfoConfig));
      }
   }

   return subFactory.create_macau_prior(session, prior_type, side_infos, config_items);
}

}
