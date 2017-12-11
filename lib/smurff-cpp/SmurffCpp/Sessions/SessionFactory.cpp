#include "SessionFactory.h"

#include <string>
#include <vector>

#include <SmurffCpp/Sessions/BaseSession.h>
#include <SmurffCpp/Sessions/CmdSession.h>
#include <SmurffCpp/Sessions/PythonSession.h>

using namespace smurff;

//create cmd session
//parses args with setFromArgs, then internally calls setFromConfig (to validate, save, set config)
std::shared_ptr<ISession> SessionFactory::create_cmd_session(int argc, char** argv)
{
   std::shared_ptr<CmdSession> session(new CmdSession());
   session->setFromArgs(argc, argv);
   return session;
}

std::shared_ptr<ISession> SessionFactory::create_cmd_session(Config& cfg)
{
   std::shared_ptr<CmdSession> session(new CmdSession());
   session->setFromConfig(cfg);
   return session;
}

//create python session
//parses args outside of c++ code (in python code)
//this is why config is passed directly from python to setFromConfig (to validate, save, set config)
std::shared_ptr<ISession> SessionFactory::create_py_session(Config& cfg)
{
   std::shared_ptr<PythonSession> session(new PythonSession());
   session->setFromConfig(cfg);
   return session;
}