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

int main(int argc, char** argv) {
    assert(argc>3 && "Usage GFA N D iter");

    int N = atoi(argv[1]);
    int D = atoi(argv[2]);
    int iter_max = atoi(argv[3]);

    assert(D>0 && N>0 && iter_max > 0 && "Usage GFA N D iter_max");

    DenseMF slave_model(num_latent);
    Macau slave_macau(slave_model);
    slave_macau.setSamples(10, iter_max);

    // fixed gaussian noise
    slave_macau.setPrecision(1.0);
    slave_macau.setVerbose(true);

    // = random_Ydense(N,D,3);
    auto Ytrain1 = ones_Ydense(N,D,2);
    auto Ytest1  = extract(Ytrain1, .2);
    slave_model.setRelationData(Ytrain1);
    slave_model.setRelationDataTest(Ytest1);

    //-- Normal priors
    //slave_macau.addPrior<DenseSpikeAndSlabPrior>(model);
    slave_macau.addPrior<DenseNormalPrior>();
    slave_macau.addPrior<SlavePrior<DenseNormalPrior>>();

    DenseMF master_model(num_latent);
    Macau master_macau(master_model);
    master_macau.setSamples(10, iter_max);

    // fixed gaussian noise
    master_macau.setPrecision(1.0);
    master_macau.setVerbose(true);

    // = random_Ydense(N,D,3);
    auto Ytrain2 = ones_Ydense(N,D,2);
    auto Ytest2  = extract(Ytrain2, .2);
    master_model.setRelationData(Ytrain2);
    master_model.setRelationDataTest(Ytest2);

    //-- Normal priors
    //master_macau.addPrior<DenseSpikeAndSlabPrior>(model);
    master_macau.addPrior<DenseNormalPrior>();
    auto &master_prior = master_macau.addPrior<MasterNormalPrior<DenseNormalPrior>>();
    master_prior.addSideInfo(slave_macau);

    master_macau.run();
}
