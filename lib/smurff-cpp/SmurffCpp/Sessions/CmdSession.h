#pragma once

#include <SmurffCpp/Sessions/Session.h>

namespace smurff {

   class CmdSession : public Session
   {
   public:
      CmdSession() {}

   public:
      void setFromArgs(int argc, char** argv);

   private:
      bool parse_options(int argc, char* argv[]);
   };

   std::shared_ptr<ISession> create_cmd_session(int argc, char** argv);
}
