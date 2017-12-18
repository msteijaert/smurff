#include "PythonSession.h"

#include <signal.h>

#include <SmurffCpp/Priors/ILatentPrior.h>

using namespace smurff;

bool PythonSession::keepRunning = true;

void PythonSession::step()
{
   if (!keepRunning)
   return;
   signal(SIGINT, intHandler);
   Session::step();
}

void PythonSession::intHandler(int)
{
   keepRunning = false;
   printf("[Received Ctrl-C. Stopping after finishing the current iteration.]\n");
}

//create python session
//parses args outside of c++ code (in python code)
//this is why config is passed directly from python to setFromConfig (to validate, save, set config)
std::shared_ptr<ISession> smurff::create_py_session(Config& cfg)
{
   std::shared_ptr<PythonSession> session(new PythonSession());
   session->setFromConfig(cfg);
   return session;
}