#pragma once

#include <memory>

#include <SmurffCpp/Sessions/CmdSessionBoost.h>

namespace smurff {

class SessionFactory;

class MPISession : public CmdSessionBoost
{
   friend class SessionFactory;

public:
   int world_rank;
   int world_size;

public:
   MPISession();

   void run();
};

std::shared_ptr<ISession> create_mpi_session(int argc, char** argv);

}
