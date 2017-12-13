#pragma once

#include <SmurffCpp/Sessions/Session.h>

namespace smurff {
   
   class SessionFactory;
   
   class CmdSession :  public Session
   {
      friend class SessionFactory;
   
   protected:
      CmdSession() {}
   
   public:
      void setFromArgs(int argc, char** argv);
   };
   
   }