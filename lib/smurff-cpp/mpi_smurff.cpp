#include <mpi.h>
#include <stdio.h>
#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <iostream>
#include <fstream>

#include <fstream>
#include <string>
#include <algorithm>
#include <random>
#include <chrono>
#include <memory>
#include <cmath>
#include <stdlib.h>

#include <unsupported/Eigen/SparseExtra>

#include "omp_util.h"
#include "linop.h"
#include "session.h"
#include "MacauOnePrior.hpp"

using namespace Eigen;
using namespace std;

int main(int argc, char** argv) 
{
    // Initialize the MPI environment
    MPI_Init(NULL, NULL);
    // Get the number of processes
    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Get the name of the processor
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);

    MPISession macau;
    macau.setFromArgs(argc, argv);

    macau.run();

    // Finalize the MPI environment.
    MPI_Finalize();
    return 0;
}

MPISession::MPISession()
{
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
}
 
void MPISession::run()
{
   if (world_rank == 0) {
       Session::run();
   } else {
       bool work_done = false;
       for(auto &p : priors) work_done |= p->run_slave();
       assert(work_done);
   }
}