#pragma once

#include <memory>

#include <SmurffCpp/Sessions/CmdSession.h>

namespace smurff {

class SessionFactory;

class MPISession : public CmdSession
{
   friend class SessionFactory;

public:
   int world_rank;
   int world_size;

protected:
   MPISession();

   void run();
};

std::shared_ptr<ISession> create_mpi_session(int argc, char** argv);

}