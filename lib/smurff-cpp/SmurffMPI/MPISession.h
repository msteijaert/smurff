#pragma once

#include <memory>

#include <SmurffCpp/Sessions/CmdSession.h>
#include <SmurffCpp/Priors/IPriorFactory.h>

namespace smurff {

class MPISession : public CmdSession
{
public:
   int world_rank;
   int world_size;

public:
   MPISession();

   void run() override;

public:
   std::shared_ptr<IPriorFactory> create_prior_factory() const override;
};

std::shared_ptr<ISession> create_mpi_session(int argc, char** argv);

}
