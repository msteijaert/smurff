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

#include "MPISession.h"

using namespace std;

/*
TEST_CASE( "utils/split_work_mpi", "Test if work splitting is correct") {
   int work3[3], work5[5];
   split_work_mpi(96, 3, work3);
   REQUIRE( work3[0] == 32 );
   REQUIRE( work3[1] == 32 );
   REQUIRE( work3[2] == 32 );

   split_work_mpi(97, 3, work3);
   REQUIRE( work3[0] == 33 );
   REQUIRE( work3[1] == 32 );
   REQUIRE( work3[2] == 32 );

   split_work_mpi(95, 3, work3);
   REQUIRE( work3[0] == 32 );
   REQUIRE( work3[1] == 32 );
   REQUIRE( work3[2] == 31 );

   split_work_mpi(80, 3, work3);
   REQUIRE( work3[0] == 28 );
   REQUIRE( work3[1] == 26 );
   REQUIRE( work3[2] == 26 );

   split_work_mpi(11, 5, work5);
   REQUIRE( work5[0] == 3 );
   REQUIRE( work5[1] == 2 );
   REQUIRE( work5[2] == 2 );
   REQUIRE( work5[3] == 2 );
   REQUIRE( work5[4] == 2 );
}
*/

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

    auto session = smurff::create_mpi_session(argc, argv);
    session->run();

    // Finalize the MPI environment.
    MPI_Finalize();
    return 0;
}
