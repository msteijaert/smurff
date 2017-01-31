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
    Macau macau;
    DenseMF &master_model = macau.denseModel(num_latent);
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
