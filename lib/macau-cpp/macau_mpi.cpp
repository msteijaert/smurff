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

#include <getopt.h>

#include <unsupported/Eigen/SparseExtra>

#include "omp_util.h"
#include "linop.h"
#include "macau.h"
#include "macauoneprior.h"

using namespace Eigen;
using namespace std;

extern "C" {
  #include <dsparse.h>
}

void usage() {
   printf("Usage:\n");
   printf("  macau_mpi --train <train_file> --row-features <feature-file> [options]\n");
   printf("Optional:\n");
   printf("  --test    test_file  test data (for computing RMSE)\n");
   printf("  --burnin        200  number of samples to discard\n");
   printf("  --nsamples      800  number of samples to collect\n");
   printf("  --num-latent     96  number of latent dimensions\n");
   printf("  --precision     5.0  precision of observations\n");
   printf("  --lambda-beta  10.0  initial value of lambda beta\n");
   printf("  --tol          1e-6  tolerance for CG\n");
   printf("  --output    results  prefix for result files\n");
}

void die(std::string message, int world_rank) {
   if (world_rank == 0) {
      std::cout << message;
   }
   MPI_Finalize();
   exit(1);
}

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

    char* fname_train         = NULL;
    char* fname_test          = NULL;
    char* fname_row_features  = NULL;
    std::string output_prefix = std::string("result");
    double precision   = 5.0;
    double lambda_beta = 10.0;
    double tol         = 1e-6;
    int burnin     = 200;
    int nsamples   = 800;
    int num_latent = 96;

    // reading command line arguments
    while (1) {
        static struct option long_options[] =
        {
            {"train",      required_argument, 0, 't'},
            {"test",       required_argument, 0, 'e'},
            {"row-features", required_argument, 0, 'r'},
            {"precision",  required_argument, 0, 'p'},
            {"burnin",     required_argument, 0, 'b'},
            {"nsamples",   required_argument, 0, 'n'},
            {"output",     required_argument, 0, 'o'},
            {"num-latent", required_argument, 0, 'l'},
            {"lambda-beta",required_argument, 0, 'a'},
            {"tol",        required_argument, 0, 'c'},
            {0, 0, 0, 0}
        };
        int option_index = 0;
        int c = getopt_long(argc, argv, "t:e:r:p:b:n:o:a:c:", long_options, &option_index);
        if (c == -1)
            break;

        switch (c) {
            case 'a': lambda_beta   = strtod(optarg, NULL); break;
            case 'b': burnin        = strtol(optarg, NULL, 10); break;
            case 'c': tol           = atof(optarg); break;
            case 'e': fname_test    = optarg; break;
            case 'l': num_latent    = strtol(optarg, NULL, 10); break;
            case 'n': nsamples      = strtol(optarg, NULL, 10); break;
            case 'o': output_prefix = std::string(optarg); break;
            case 'p': precision     = strtod(optarg, NULL); break;
            case 'r': fname_row_features = optarg; break;
            case 't': fname_train = optarg; break;
            case '?':
            default:
                      if (world_rank == 0)
                          usage();
                      MPI_Finalize();
                      exit(1);
        }
    }
    if (fname_train == NULL || fname_row_features == NULL) {
        if (world_rank == 0) {
            printf("[ERROR]\nMissing parameters '--matrix' or '--row-features'.\n");
            usage();
        }
        MPI_Finalize();
        exit(1);
    }
    if (world_rank == 0) {
        printf("Train data:    '%s'\n", fname_train);
        printf("Test data:     '%s'\n", fname_test==NULL ?"" :fname_test);
        printf("Row features:  '%s'\n", fname_row_features);
        printf("Output prefix: '%s'\n", output_prefix.c_str());
        printf("Burn-in:       %d\n", burnin);
        printf("Samples:       %d\n", nsamples);
        printf("Num-latents:   %d\n", num_latent);
        printf("Precision:     %.1f\n", precision);
        printf("Lambda-beta:   %.1f\n", lambda_beta);
        printf("tol:           %.1e\n", tol);
    }
    if ( ! file_exists(fname_train) ) {
        die(std::string("[ERROR]\nTrain data file '") + fname_train + "' not found.\n", world_rank);
    }
    if ( ! file_exists(fname_row_features) ) {
        die(std::string("[ERROR]\nRow feature file '") + fname_row_features + "' not found.\n", world_rank);
    }
    if ( (fname_test != NULL) && ! file_exists(fname_test) ) {
        die(std::string("[ERROR]\nTest data file '") + fname_test + "' not found.\n", world_rank);
    }

    // Step 1. Loading data
    //std::unique_ptr<SparseFeat> row_features = load_bcsr(fname_row_features);
    auto row_features = load_bcsr(fname_row_features);
    if (world_rank == 0) {
        printf("Row features:   [%d x %d].\n", row_features->rows(), row_features->cols());
    }
    SparseDoubleMatrix* Y     = NULL;
    SparseDoubleMatrix* Ytest = NULL;

    SparseMF model(num_latent);
    MacauMPI macau(model);
    macau.setSamples(burnin, nsamples);

    // -- noise model + general parameters
    macau.setPrecision(precision);

    macau.setVerbose(true);
    Y = read_sdm(fname_train);
    model.setRelationData(*Y);

    //-- Normal column prior
    //macau.addPrior<SparseNormalPrior>(model);
    macau.addPrior<SpikeAndSlabPrior>(model);

    //-- row prior with side information
    auto &prior_u = macau.addPrior<MacauOnePrior<SparseFeat>>(model);
    prior_u.addSideInfo(row_features, false);
    prior_u.setLambdaBeta(lambda_beta);
    //prior_u.setTol(tol);

    // test data
    if (fname_test != NULL) {
        Ytest = read_sdm(fname_test);
        macau.model.setRelationDataTest(*Ytest);
    }

    if (world_rank == 0) {
        printf("Training data:  %ld [%d x %d]\n", Y->nnz, Y->nrow, Y->ncol);
        if (Ytest != NULL) {
            printf("Test data:      %ld [%d x %d]\n", Ytest->nnz, Ytest->nrow, Ytest->ncol);
        } else {
            printf("Test data:      --\n");
        }
    }

    delete Y;
    if (Ytest) delete Ytest;

    macau.run();

    // Finalize the MPI environment.
    MPI_Finalize();
    return 0;
}

MacauMPI::MacauMPI(Factors &m)
    : Macau(m)
{
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
}
 

void MacauMPI::run()
{
   /* adapted from Macau.run() */
   init();
   if (world_rank == 0) {
       Macau::run();
   } else {
       bool work_done = false;
       for(auto &p : priors) work_done |= p->run_slave();
       assert(work_done);
   }
}

template<class FType>
MacauMPIPrior<FType>::MacauMPIPrior(SparseMF &m, int p, INoiseModel &n) 
 : MacauPrior<FType>(m, p, n)
{
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
}


template<class FType>
void MacauMPIPrior<FType>::addSideInfo(std::unique_ptr<FType> &Fmat, bool comp_FtF)
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
void MacauMPIPrior<FType>::sample_beta() {
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

