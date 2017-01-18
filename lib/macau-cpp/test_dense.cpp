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

const int num_latent = 32;

int main(int argc, char** argv) {
    assert(argc>3 && "Usage GFA N D iter");

    int N = atoi(argv[1]);
    int D = atoi(argv[2]);
    int iter_max = atoi(argv[3]);

    assert(D>0 && N>0 && iter_max > 0 && "Usage GFA N D iter_max");

    DenseMF model(num_latent);
    Macau macau(model);
    macau.setSamples(iter_max / 5, 4 * iter_max / 5);

    // fixed gaussian noise
    macau.setPrecision(1.0);
    macau.setVerbose(true);

    // = random_Ydense(N,D,3);
    auto Ytrain = ones_Ydense(N,D,2);
    auto Ytest  = extract(Ytrain, .2);
    model.setRelationData(Ytrain);
    model.setRelationDataTest(Ytest);

    //-- Normal priors
    macau.addPrior<DenseNormalPrior>(model);
    macau.addPrior<DenseNormalPrior>(model);

    macau.run();
}
