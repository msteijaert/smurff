#pragma once

#include <memory>

#include <SmurffCpp/Sessions/CmdSession.h>

namespace smurff {

class MPISession : public CmdSession
{
public:
   int world_rank;
   int world_size;

public:
   MPISession();

   void run();
};

std::shared_ptr<ISession> create_mpi_session(int argc, char** argv);

}
