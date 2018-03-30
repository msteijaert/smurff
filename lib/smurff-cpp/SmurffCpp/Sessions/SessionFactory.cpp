#include "SessionFactory.h"

#include <string>
#include <vector>

#include <SmurffCpp/Sessions/PythonSession.h>

using namespace smurff;

//create python session
//parses args outside of c++ code (in python code)
//this is why config is passed directly from python to setFromConfig (to validate, save, set config)
std::shared_ptr<ISession> SessionFactory::create_py_session(Config& cfg)
{
   std::shared_ptr<PythonSession> session(new PythonSession());
   session->setCreateFromConfig(cfg);
   return session;
}

std::shared_ptr<ISession> SessionFactory::create_py_session(const std::string& rootPath)
{
   std::shared_ptr<PythonSession> session(new PythonSession());
   session->setRestoreFromRootPath(rootPath);
   return session;
}

//for testing only
std::shared_ptr<ISession> SessionFactory::create_session(Config& cfg)
{
   std::shared_ptr<Session> session(new Session());
   session->setCreateFromConfig(cfg);
   return session;
}
