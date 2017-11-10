#pragma once

#include <memory>

#include <SmurffCpp/Sessions/Session.h>
#include <SmurffCpp/Sessions/ISession.h>

namespace smurff {

   class SessionFactory
   {
   public:
      static std::shared_ptr<ISession> create_cmd_session(int argc, char** argv);

      static std::shared_ptr<ISession> create_py_session(Config& cfg);
   };

}