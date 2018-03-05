#pragma once

#include <memory>
#include <string>

#include <SmurffCpp/Configs/Config.h>
#include <SmurffCpp/Sessions/ISession.h>

namespace smurff {

   class SessionFactory
   {
   public:
      static std::shared_ptr<ISession> create_py_session(Config& cfg);
      static std::shared_ptr<ISession> create_py_session(const std::string& rootPath);
   };

}