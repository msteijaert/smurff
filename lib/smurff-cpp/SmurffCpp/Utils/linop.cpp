#include <cmath>
#include <iostream>
#include <stdexcept>

//#define EIGEN_USE_BLAS

#include <Eigen/Dense>

#include <SmurffCpp/Utils/Error.h>

#include "linop.h"
#include "omp_util.h"

using namespace Eigen;
using namespace std;

void smurff::linop::makeSymmetric(Eigen::MatrixXd & A) {
  A = A.selfadjointView<Eigen::Lower>();
}
