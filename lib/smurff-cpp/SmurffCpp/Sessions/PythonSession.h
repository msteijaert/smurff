#pragma once

#include <SmurffCpp/Sessions/Session.h>

namespace smurff {

class SessionFactory;

class PythonSession : public Session
{
   friend class SessionFactory;

private:
   static bool keepRunning;
   static bool keepRunningVerbose;

protected:
   PythonSession()
   {
      name = "PythonSession";
      keepRunning = true;
   }

protected:
   bool interrupted() override
   {
       return !keepRunning;
   }

   bool step() override;

private:
   static void intHandler(int);
};

}
