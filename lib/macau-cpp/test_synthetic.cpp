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

#include "session.h"
#include "gen_random.h"

using namespace Eigen;
using namespace std;
using namespace Macau;

const int num_latent = 5;



void test_sparse(int N, int D, int iter_max) 
{
    assert(D>0 && N>0 && iter_max > 0 && "Usage GFA N D iter_max");
    Session macau;
    auto &master_model = macau.sparseModel(num_latent);
    macau.burnin = 10;
    macau.nsamples = iter_max;
    macau.verbose = true;

    // fixed gaussian noise
    macau.setPrecision(1.0);

    // = random_Ydense(N,D,3);
    auto Ytrain = ones_Ysparse(N,D,2,.8);
    auto predictions = extract(Ytrain,.2);
    master_model.setRelationData(Ytrain);
    macau.pred.set(predictions);

    //-- Normal priors
    macau.addPrior<SpikeAndSlabPrior>();
    //macau.addPrior<NormalPrior>();
    auto &master_prior = macau.addPrior<MasterPrior<NormalPrior>>();
    auto &slave_model = master_prior.addSlave<SparseMF>();
    auto Y1 = ones_Ysparse(N,2*D,2,.3);
    slave_model.setRelationData(Y1);

    macau.run();
}

void test_dense_dense(int N, int D, int iter_max) 
{
    assert(D>0 && N>0 && iter_max > 0 && "Usage GFA N D iter_max");
    Session macau;
    auto &master_model = macau.denseDenseModel(num_latent);
    macau.burnin = 10;
    macau.nsamples = iter_max;
    macau.verbose = true;

    // fixed gaussian noise
    macau.setPrecision(1.0);

    // = random_Ydense(N,D,3);
    auto Ytrain2 = ones_Ydense(N,D,2);
    auto predictions2  = extract(Ytrain2, .2);
    master_model.setRelationData(Ytrain2);
    macau.pred.set(predictions2);

    //-- Normal priors
    macau.addPrior<SpikeAndSlabPrior>();
    //macau.addPrior<NormalPrior>();
    //macau.addPrior<NormalPrior>();
    auto &master_prior = macau.addPrior<MasterPrior<NormalPrior>>();
    auto &slave_model = master_prior.addSlave<DenseDenseMF>();
    auto Ytrain1 = ones_Ydense(N,2*D,2);
    slave_model.setRelationData(Ytrain1);

    macau.run();
}

int main(int argc, char** argv) {
    assert(argc>4 && "Usage test N D iter <dense_dense|sparse>");

    int N = atoi(argv[1]);
    int D = atoi(argv[2]);
    int iter_max = atoi(argv[3]);
    if (!strcmp(argv[4], "dense_dense")) test_dense_dense(N, D, iter_max);
    else if (!strcmp(argv[4], "sparse")) test_sparse(N, D, iter_max);
    else assert(false);

    return 0;
}


