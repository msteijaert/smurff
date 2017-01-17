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

    //MatrixXd Y = random_Ydense(N,D,3);
    MatrixXd Y = ones_Ydense(N,D,2);
    Macau macau(num_latent);

    // fixed gaussian noise
    macau.setPrecision(1.0);
    macau.setVerbose(true);
    macau.model.setRelationData(Y);

   //-- Normal priors
   macau.addPrior<DenseNormalPrior>();
   macau.addPrior<DenseNormalPrior>();

   macau.model.init();

   macau.run();
}
