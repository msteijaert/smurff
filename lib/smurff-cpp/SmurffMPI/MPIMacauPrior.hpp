#pragma once

#include <memory>

#include <mpi.h>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <Utils/Distribution.h>

#include "MacauPrior.h"

#include "linop.h"
#include "Model.h"
#include "data.h"

namespace smurff 
{

//why not use init method ?

// Prior with side information
template<class FType>
class MPIMacauPrior : public MacauPrior<FType> 
{
public:
   int world_rank;
   int world_size;

private:
   int* rhs_for_rank = NULL;
   double* rec     = NULL;
   int* sendcounts = NULL;
   int* displs     = NULL;   

public:
   //TODO: missing implementation
   MPIMacauPrior(BaseSession& session, int mode);

   MPIMacauPrior(ScarceMatrixData& m, int mode, INoiseModel &n) 
    : MacauPrior<FType>(m, mode, n)
   {
      MPI_Comm_size(MPI_COMM_WORLD, &world_size);
      MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
   }

   virtual ~MPIMacauPrior() {}

   void addSideInfo(std::unique_ptr<FType> &Fmat, bool comp_FtF = false)
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

   //TODO: missing implementation
   std::ostream &info(std::ostream &os, std::string indent) override;

   virtual void sample_beta()
   {
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
      for (int f = 0; f < num_feat; f++) 
      {
         for (int d = 0; d < nrhs; d++) 
         {
            RHS(d, f) = rec[f + d * num_feat];
         }
      }
      // solving
      solve_blockcg(result, *this->F, this->lambda_beta, RHS, this->tol, 32, 8);
      result.transposeInPlace();
      MPI_Gatherv(result.data(), nrhs*num_feat, MPI_DOUBLE, this->Ft_y.data(), sendcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
      if (world_rank == 0) 
      {
         //this->beta = Ft_y.transpose();
         #pragma omp parallel for schedule(static)
         for (int f = 0; f < num_feat; f++) 
         {
            for (int d = 0; d < num_latent; d++) 
            {
               this->beta(d, f) = this->Ft_y(f, d);
            }
         }
      }
   }

   virtual bool run_slave()
   {
      sample_beta(); 
      return true; 
   }

   int rhs() const 
   {
      return rhs_for_rank[world_rank]; 
   }

   void split_work_mpi(int num_latent, int num_nodes, int* work) 
   {
      double avg_work = num_latent / (double) num_nodes;
      int work_unit;
      if (2 <= avg_work) work_unit = 2;
      else work_unit = 1;
   
      int min_work  = work_unit * (int)floor(avg_work / work_unit);
      int work_left = num_latent;
   
      for (int i = 0; i < num_nodes; i++) {
         work[i]    = min_work;
         work_left -= min_work;
      }
      int i = 0;
      while (work_left > 0) {
         int take = std::min(work_left, work_unit);
         work[i]   += take;
         work_left -= take;
         i = (i + 1) % num_nodes;
      }
   }
};

}