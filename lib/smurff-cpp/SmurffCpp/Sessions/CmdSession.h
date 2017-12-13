#pragma once

#include <SmurffCpp/Sessions/Session.h>

namespace smurff {
   
   class SessionFactory;
   
   class CmdSessionBoost :  public Session
   {
      friend class SessionFactory;
   
   protected:
      CmdSessionBoost() {}
   
   public:
      void setFromArgs(int argc, char** argv);
   };
   
   }