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
#include "macau.h"
#include "macauoneprior.h"

using namespace Eigen;
using namespace std;

int main(int argc, char** argv) {
    int num_latent = 32;
    SparseMF model(num_latent);
    Macau macau(model);
    assert(macau.setFromArgs(argc, argv, model, true));

    macau.run();
    return 0;
}
