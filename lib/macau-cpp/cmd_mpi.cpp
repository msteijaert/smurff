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
#include "iface.h"
#include "macauoneprior.h"

using namespace Eigen;
using namespace std;

int main(int argc, char** argv) {
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

template<class FType>
MPIMacauPrior<FType>::MPIMacauPrior(SparseMF &m, int p, INoiseModel &n) 
 : MacauPrior<FType>(m, p, n)
{
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
}


template<class FType>
void MPIMacauPrior<FType>::addSideInfo(std::unique_ptr<FType> &Fmat, bool comp_FtF)
{
    MacauPrior<FType>::addSideInfo(Fmat, comp_FtF);

    rhs_for_rank = new int[world_size];
    split_work_mpi(this->num_latent(), world_size, rhs_for_rank);

    sendcounts = new int[world_size];
    displs     = new int[world_size];
    int sum = 0;
    for (int n = 0; n < world_size; n++) {
        sendcounts[n] = rhs_for_rank[n] * Fmat->cols();
        displs[n]     = sum;
        sum          += sendcounts[n];
    }
    rec = new double[sendcounts[world_rank]];
}


template<class FType>
void MPIMacauPrior<FType>::sample_beta() {
   const int num_latent = this->beta.rows();
   const int num_feat = this->beta.cols();

   if (world_rank == 0) {
      this->Ft_y = this->compute_Ft_y_omp();
      this->Ft_y.transposeInPlace();
   }

   MPI_Bcast(& this->lambda_beta, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
   MPI_Bcast(& this->tol,         1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

   // sending Ft_y
   MPI_Scatterv(this->Ft_y.data(), sendcounts, displs, MPI_DOUBLE, rec, sendcounts[world_rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
   int nrhs = rhs_for_rank[world_rank];
   MatrixXd RHS(nrhs, num_feat), result(nrhs, num_feat);

#pragma omp parallel for schedule(static)
   for (int f = 0; f < num_feat; f++) {
      for (int d = 0; d < nrhs; d++) {
         RHS(d, f) = rec[f + d * num_feat];
      }
   }
   // solving
   solve_blockcg(result, *this->F, this->lambda_beta, RHS, this->tol, 32, 8);
   result.transposeInPlace();
   MPI_Gatherv(result.data(), nrhs*num_feat, MPI_DOUBLE, this->Ft_y.data(), sendcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
   if (world_rank == 0) {
      //this->beta = Ft_y.transpose();
#pragma omp parallel for schedule(static)
      for (int f = 0; f < num_feat; f++) {
         for (int d = 0; d < num_latent; d++) {
            this->beta(d, f) = this->Ft_y(f, d);
         }
      }
   }
}

