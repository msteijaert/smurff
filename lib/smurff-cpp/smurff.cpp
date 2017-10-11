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

#include <Priors/ILatentPrior.h>

#include <Sessions/CmdSession.h>

using namespace Eigen;
using namespace std;
using namespace smurff;

int main(int argc, char** argv) {
    CmdSession session;
    session.setFromArgs(argc, argv);
    session.run();
    return 0;
}
