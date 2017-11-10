#pragma once

#include <SmurffCpp/Sessions/Session.h>

namespace smurff {

class SessionFactory;

class PythonSession : public Session
{
   friend class SessionFactory;

private:
   static bool keepRunning;

protected:
   PythonSession()
   {
      name = "PythonSession";
      keepRunning = true;
   }

protected:
   void step() override;

private:
   static void intHandler(int);
};

}