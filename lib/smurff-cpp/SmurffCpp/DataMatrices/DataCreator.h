#pragma once

#include <memory>

#include "IDataCreator.h"

#include <SmurffCpp/Sessions/Session.h>

namespace smurff
{
   class DataCreator : public IDataCreator
   {
   private:
      std::shared_ptr<Session> m_session;

   public:
      DataCreator(std::shared_ptr<Session> session)
         : m_session(session)
      {
      }

   public:
      std::shared_ptr<Data> create(std::shared_ptr<const MatrixConfig> mc) const override;
      std::shared_ptr<Data> create(std::shared_ptr<const TensorConfig> tc) const override;
   };
}