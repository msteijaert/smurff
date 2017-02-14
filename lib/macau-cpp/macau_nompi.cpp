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
using namespace Macau;

int main(int argc, char** argv) {
    Session macau;
    macau.setFromArgs(argc, argv, true);
    macau.run();
    return 0;
}
