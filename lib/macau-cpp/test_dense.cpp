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

#include "macau.h"
#include "gen_random.h"

using namespace Eigen;
using namespace std;

const int num_latent = 5;



void test_sparse(int N, int D, int iter_max) 
{
    assert(D>0 && N>0 && iter_max > 0 && "Usage GFA N D iter_max");
    Macau macau;
    auto &master_model = macau.sparseModel(num_latent);
    macau.setSamples(10, iter_max);

    // fixed gaussian noise
    macau.setPrecision(1.0);
    macau.setVerbose(true);

    // = random_Ydense(N,D,3);
    auto Y2 = split(ones_Ysparse(N,D,2,.3),.2);
    master_model.setRelationData(Y2.first);
    master_model.setRelationDataTest(Y2.second);

    //-- Normal priors
    //macau.addPrior<DenseSpikeAndSlabPrior>(model);
    macau.addPrior<NormalPrior>();
    //macau.addPrior<NormalPrior>();
    auto &master_prior = macau.addPrior<MasterPrior<NormalPrior>>();
    auto &slave_model = master_prior.addSlave<SparseMF>();
    auto Y1 = ones_Ysparse(N,2*D,2,.3);
    slave_model.setRelationData(Y1);

    macau.run();
}

void test_dense(int N, int D, int iter_max) 
{
    assert(D>0 && N>0 && iter_max > 0 && "Usage GFA N D iter_max");
    Macau macau;
    auto &master_model = macau.denseModel(num_latent);
    macau.setSamples(10, iter_max);

    // fixed gaussian noise
    macau.setPrecision(1.0);
    macau.setVerbose(true);

    // = random_Ydense(N,D,3);
    auto Ytrain2 = ones_Ydense(N,D,2);
    auto Ytest2  = extract(Ytrain2, .2);
    master_model.setRelationData(Ytrain2);
    master_model.setRelationDataTest(Ytest2);

    //-- Normal priors
    //macau.addPrior<DenseSpikeAndSlabPrior>(model);
    macau.addPrior<NormalPrior>();
    //macau.addPrior<NormalPrior>();
    auto &master_prior = macau.addPrior<MasterPrior<NormalPrior>>();
    auto &slave_model = master_prior.addSlave<DenseMF>();
    auto Ytrain1 = ones_Ydense(N,2*D,2);
    slave_model.setRelationData(Ytrain1);

    macau.run();
}

int main(int argc, char** argv) {
    assert(argc>4 && "Usage test N D iter <dense|sparse>");

    int N = atoi(argv[1]);
    int D = atoi(argv[2]);
    int iter_max = atoi(argv[3]);
    bool dense = !strcmp(argv[4], "dense");
    if (dense) test_dense(N, D, iter_max);
    else test_sparse(N, D, iter_max);

    return 0;
}


