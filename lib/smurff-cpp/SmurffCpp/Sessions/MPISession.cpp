#include <mpi.h>
#include <assert.h>

#include <SmurffCpp/Priors/ILatentPrior.h>
#include "MPISession.h"

using namespace smurff;

MPISession::MPISession()
{
   name = "MPISession"; 

   MPI_Comm_size(MPI_COMM_WORLD, &world_size);
   MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
}

void MPISession::run()
{
   if (world_rank == 0) 
   {
      Session::run();
   } 
   else 
   {
      bool work_done = false;

      for(auto &p : m_priors) 
         work_done |= p->run_slave();
         
      assert(work_done);
   }
}