#include "SessionFactory.h"

#include <string>
#include <vector>

#include <SmurffCpp/Sessions/BaseSession.h>
#include <SmurffCpp/Sessions/CmdSession.h>
#include <SmurffCpp/Sessions/PythonSession.h>

#ifdef MPI_FOUND
#include <SmurffCpp/Sessions/MPISession.h>
#endif

using namespace smurff;

//create cmd session
//parses args with setFromArgs, then internally calls setFromConfig (to validate, save, set config)
std::shared_ptr<ISession> SessionFactory::create_cmd_session(int argc, char** argv)
{
   std::shared_ptr<CmdSession> session(new CmdSession());
   session->setFromArgs(argc, argv);
   return session;
}

#ifdef MPI_FOUND

//create mpi session
//parses args with setFromArgs, then internally calls setFromConfig (to validate, save, set config)
std::shared_ptr<ISession> SessionFactory::create_mpi_session(int argc, char** argv)
{
   std::shared_ptr<MPISession> session(new MPISession());
   session->setFromArgs(argc, argv);
   return session;
}

#endif

//create python session
//parses args outside of c++ code (in python code)
//this is why config is passed directly from python to setFromConfig (to validate, save, set config)
std::shared_ptr<ISession> SessionFactory::create_py_session(Config& cfg)
{
   std::shared_ptr<PythonSession> session(new PythonSession());
   session->setFromConfig(cfg);
   return session;
}