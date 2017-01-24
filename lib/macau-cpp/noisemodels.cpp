#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <memory>
#include <cmath>

#include "mvnormal.h"
#include "noisemodels.h"
#include "model.h"

using namespace Eigen;

//  AdaptiveGaussianNoise  ////
void AdaptiveGaussianNoise::init() {
    double var_total = model.var_total();
 
    // Var(noise) = Var(total) / (SN + 1)
    alpha     = (sn_init + 1.0) / var_total;
    alpha_max = (sn_max + 1.0) / var_total;
}

void AdaptiveGaussianNoise::update()
{
    double sumsq = model.sumsq();

    // (a0, b0) correspond to a prior of 1 sample of noise with full variance
    double a0 = 0.5;
    double b0 = 0.5 * var_total;
    double aN = a0 + model.Ynnz() / 2.0;
    double bN = b0 + sumsq / 2.0;
    alpha = rgamma(aN, 1.0 / bN);
    if (alpha > alpha_max) {
        alpha = alpha_max;
    }
}
