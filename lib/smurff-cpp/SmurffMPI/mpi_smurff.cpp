#include <mpi.h>
#include <stdio.h>

#include <iostream>
#include <fstream>

#include <fstream>
#include <string>
#include <algorithm>
#include <chrono>
#include <memory>
#include <cmath>
#include <stdlib.h>

#include <SmurffCpp/Utils/omp_util.h>
#include <SmurffCpp/Priors/ILatentPrior.h>

#include <SmurffCpp/Sessions/SessionFactory.h>

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

    std::shared_ptr<smurff::MPISession> session = SessionFactory::create_mpi_session(argc, argv);
    session->run();

    // Finalize the MPI environment.
    MPI_Finalize();
    return 0;
}
